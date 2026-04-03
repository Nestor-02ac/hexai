"""
Gumbel AlphaZero search for Y.
"""

import importlib
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from y_board import Player
from neural_net import encode_board


def _masked_softmax(logits):
    finite_mask = np.isfinite(logits)
    if not finite_mask.any():
        return np.zeros_like(logits, dtype=np.float32)
    max_val = logits[finite_mask].max()
    exp = np.zeros_like(logits, dtype=np.float32)
    exp[finite_mask] = np.exp(logits[finite_mask] - max_val)
    total = exp.sum()
    return exp / total if total > 0 else exp


class Node:
    __slots__ = [
        "action",
        "action_player",
        "prior",
        "policy_logit",
        "policy_noise",
        "value",
        "reward",
        "mean",
        "count",
        "children",
    ]

    def __init__(self, action=-1, action_player=0, prior=0.0, policy_logit=0.0):
        self.action = action
        self.action_player = action_player
        self.prior = prior
        self.policy_logit = policy_logit
        self.policy_noise = 0.0
        self.value = 0.0
        self.reward = 0.0
        self.mean = 0.0
        self.count = 0
        self.children = {}

    @property
    def expanded(self):
        return bool(self.children)

    def add(self, value):
        self.count += 1
        self.mean += (value - self.mean) / self.count

    def normalized_mean(self, config):
        value = self.reward + config.reward_discount * self.mean
        if self.action_player == config.value_flipping_player:
            value = -value
        return value

    def puct_score(self, total_simulation, config, init_q_value):
        puct_bias = config.pb_c_init + math.log((1 + total_simulation + config.pb_c_base) / config.pb_c_base)
        value_u = (puct_bias * self.prior * math.sqrt(max(1, total_simulation))) / (1 + self.count)
        value_q = init_q_value if self.count == 0 else self.normalized_mean(config)
        return value_u + value_q


@dataclass
class RootSearchState:
    root: Node
    player: int
    root_value: float = 0.0
    legal_actions: list[int] = field(default_factory=list)
    candidates: list[Node] = field(default_factory=list)
    sample_size: int = 0
    simulation_budget: int = 0


@dataclass
class LeafRequest:
    search: RootSearchState
    node: Node
    board: object
    to_play: int
    path: list[Node]


class GumbelMCTS:
    def __init__(self, config, network, device):
        self.config = config
        self.network = network
        self.device = device

    def evaluate_states(self, state_tensors):
        batch = torch.stack(state_tensors).to(self.device)
        with torch.no_grad():
            policy_logits, values = self.network(batch)
        return (
            policy_logits.cpu().numpy().astype(np.float32, copy=False),
            values.squeeze(-1).cpu().numpy().astype(np.float32, copy=False),
        )

    def new_search(self, player):
        return RootSearchState(
            root=Node(action=-1, action_player=3 - player),
            player=player,
        )

    def prepare_expand(self, node, board, to_play):
        return encode_board(board, to_play), board.get_empty_cells()

    def run(
        self,
        board,
        player,
        add_noise=False,
        select_action_by_count=None,
        select_action_by_softmax_count=None,
        temperature=None,
        value_threshold=None,
    ):
        search = self.new_search(player)
        state, legal_actions = self.prepare_expand(search.root, board, player)
        policy_batch, value_batch = self.evaluate_states([state])

        self.finish_root(
            search,
            policy_batch[0],
            float(value_batch[0]),
            legal_actions,
            add_noise=add_noise,
        )

        while not self.search_complete(search):
            leaf_request = self.simulate_until_leaf(search, board)
            if leaf_request is None:
                continue

            leaf_state, leaf_legal = self.prepare_expand(
                leaf_request.node,
                leaf_request.board,
                leaf_request.to_play,
            )

            policy_batch, value_batch = self.evaluate_states([leaf_state])

            self.finish_leaf(
                leaf_request,
                policy_batch[0],
                float(value_batch[0]),
                leaf_legal,
            )

        return self.finalize_search(
            search,
            select_action_by_count=select_action_by_count,
            select_action_by_softmax_count=select_action_by_softmax_count,
            temperature=temperature,
            value_threshold=value_threshold,
        )

    def finish_root(self, search, policy_logits_np, value, legal_actions, add_noise=False):
        search.root_value = self._expand_from_logits(
            search.root,
            search.player,
            policy_logits_np,
            value,
            legal_actions,
            add_noise=add_noise,
        )
        self._backup([search.root], search.root_value)
        search.legal_actions = list(search.root.children.keys())
        self._initialize_candidates(search)
        return search.root_value

    def simulate_until_leaf(self, search, board):
        if self.search_complete(search):
            return None

        candidate = self._select_root_candidate(search)
        if candidate is None:
            return None

        scratch = board.clone()
        path = [search.root, candidate]

        scratch.play(candidate.action, search.player)
        if scratch.check_win(search.player):
            self._backup(path, self._terminal_value(search.player))
            self._advance_search(search)
            return None

        current = 3 - search.player
        node = candidate

        while node.expanded:
            node = self._select_child_by_puct(node)
            scratch.play(node.action, current)
            path.append(node)

            if scratch.check_win(current):
                self._backup(path, self._terminal_value(current))
                self._advance_search(search)
                return None

            current = 3 - current

        return LeafRequest(search, node, scratch, current, path)

    def finish_leaf(self, leaf_request, policy_logits_np, value, legal_actions):
        expanded_value = self._expand_from_logits(
            leaf_request.node,
            leaf_request.to_play,
            policy_logits_np,
            value,
            legal_actions,
        )
        self._backup(leaf_request.path, expanded_value)
        self._advance_search(leaf_request.search)
        return expanded_value

    def search_complete(self, search):
        if not search.legal_actions:
            return True
        if len(search.legal_actions) <= 1:
            return True
        return search.root.count >= self.config.num_simulations + 1

    def finalize_search(self, search, **kwargs):
        policy = np.zeros(self.config.action_space, dtype=np.float32)
        if not search.legal_actions:
            return 0, policy
        if len(search.legal_actions) == 1:
            policy[search.legal_actions[0]] = 1.0
            return search.legal_actions[0], policy

        child = self._decide_action_child(search.root, search.candidates, **kwargs)
        return child.action, self._improved_policy(search.root, search.player)

    @staticmethod
    def _terminal_value(winner):
        return 1.0 if winner == Player.BLACK else -1.0


class GumbelZeroAgent:
    def __init__(self, config, network, device):
        self.mcts = create_gumbel_mcts(config, network, device)

    def select_move(self, board, player, **kwargs):
        action, _ = self.mcts.run(board, int(player), **kwargs)
        return action


def create_gumbel_mcts(config, network, device):
    backend = getattr(config, "mcts_backend", "auto")
    if backend in {"auto", "cython"}:
        try:
            module = importlib.import_module("cgumbel_mcts_y")
            return module.CGumbelMCTS(config, network, device)
        except Exception:
            if backend == "cython":
                raise
    return GumbelMCTS(config, network, device)