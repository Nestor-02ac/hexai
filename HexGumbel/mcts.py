"""Gumbel AlphaZero search for Hex."""

import importlib
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from hex_board import Player
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

    def new_search(self, player):
        return RootSearchState(
            root=Node(action=-1, action_player=3 - player),
            player=player,
        )

    def prepare_expand(self, node, board, to_play):
        return encode_board(board, to_play), board.get_empty_cells()

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

        return LeafRequest(
            search=search,
            node=node,
            board=scratch,
            to_play=current,
            path=path,
        )

    def finish_leaf(self, leaf_request, policy_logits_np, value, legal_actions):
        expanded_value = self._expand_from_logits(
            leaf_request.node,
            leaf_request.to_play,
            policy_logits_np,
            value,
            legal_actions,
            add_noise=False,
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

    def finalize_search(
        self,
        search,
        select_action_by_count=None,
        select_action_by_softmax_count=None,
        temperature=None,
        value_threshold=None,
    ):
        policy = np.zeros(self.config.action_space, dtype=np.float32)
        if not search.legal_actions:
            return 0, policy
        if len(search.legal_actions) == 1:
            policy[search.legal_actions[0]] = 1.0
            return search.legal_actions[0], policy

        child = self._decide_action_child(
            search.root,
            search.candidates,
            select_action_by_count=select_action_by_count,
            select_action_by_softmax_count=select_action_by_softmax_count,
            temperature=temperature,
            value_threshold=value_threshold,
        )
        return child.action, self._improved_policy(search.root, search.player)

    def _expand_from_logits(self, node, to_play, policy_logits_np, value, legal_actions, add_noise=False):
        policy_logits_np = np.asarray(policy_logits_np, dtype=np.float32).reshape(-1)
        node.value = float(value)
        node.children = {}

        masked_logits = np.full(self.config.action_space, -np.inf, dtype=np.float32)
        for action in legal_actions:
            masked_logits[action] = policy_logits_np[action]
        priors = _masked_softmax(masked_logits)

        for action in legal_actions:
            node.children[action] = Node(
                action=action,
                action_player=to_play,
                prior=float(priors[action]),
                policy_logit=float(policy_logits_np[action]),
            )

        if add_noise and node.children:
            noise = np.random.gumbel(size=len(node.children))
            for child, gumbel in zip(node.children.values(), noise):
                child.policy_noise = float(gumbel)
                child.policy_logit += float(gumbel)

        return node.value

    def _select_root_candidate(self, search):
        if not search.candidates:
            return None
        return min(search.candidates, key=lambda node: (node.count, -node.policy_logit))

    def _select_child_by_puct(self, node):
        total_simulation = max(0, node.count - 1)
        init_q_value = self._calculate_init_q_value(node)
        best_child = None
        best_score = -float("inf")
        best_prior = -float("inf")
        for child in node.children.values():
            score = child.puct_score(total_simulation, self.config, init_q_value)
            if score > best_score or (score == best_score and child.prior > best_prior):
                best_score = score
                best_prior = child.prior
                best_child = child
        return best_child

    def _calculate_init_q_value(self, node):
        visited = [child.normalized_mean(self.config) for child in node.children.values() if child.count > 0]
        if not visited:
            return -1.0
        return (sum(visited) - 1.0) / (len(visited) + 1.0)

    def _backup(self, path, value):
        updated_value = value
        path[-1].value = value
        for node in reversed(path):
            node.add(updated_value)
            updated_value = node.reward + self.config.reward_discount * updated_value

    def _initialize_candidates(self, search):
        search.candidates = sorted(
            search.root.children.values(),
            key=lambda node: node.policy_logit,
            reverse=True,
        )
        if len(search.candidates) > self.config.gumbel_sample_size:
            search.candidates = search.candidates[:self.config.gumbel_sample_size]
        search.sample_size = self.config.gumbel_sample_size
        if len(search.candidates) <= 1:
            search.simulation_budget = self.config.num_simulations
        else:
            search.simulation_budget = self._initial_budget(search.sample_size)

    def _advance_search(self, search):
        if len(search.candidates) <= 1 or search.sample_size <= 2:
            return
        for node in search.candidates:
            if node.count < search.simulation_budget:
                return

        next_budget = self._next_phase_budget(search.sample_size)
        if next_budget <= 0 or search.sample_size <= 2:
            return

        search.sample_size //= 2
        self._sort_candidates_by_score(search.root, search.candidates)
        if len(search.candidates) > search.sample_size:
            search.candidates = search.candidates[:search.sample_size]
        if search.candidates:
            search.simulation_budget = search.candidates[0].count + next_budget

    def _initial_budget(self, sample_size):
        if sample_size <= 1:
            return self.config.num_simulations
        log_sample_size = math.log2(self.config.gumbel_sample_size)
        denom = log_sample_size * sample_size
        if denom <= 0:
            return self.config.num_simulations
        return max(1, int(math.floor(self.config.num_simulations / denom)))

    def _next_phase_budget(self, sample_size):
        if sample_size <= 1:
            return 0
        log_sample_size = math.log2(self.config.gumbel_sample_size)
        denom = log_sample_size * sample_size / 2.0
        if denom <= 0:
            return 0
        return int(math.floor(self.config.num_simulations / denom))

    def _sort_candidates_by_score(self, root, candidates):
        max_child_count = max((child.count for child in root.children.values()), default=0)
        min_value = -float("inf")

        def score(child):
            if child.count == 0:
                return min_value
            value = child.normalized_mean(self.config)
            scale = (self.config.gumbel_sigma_visit_c + max_child_count) * self.config.gumbel_sigma_scale_c
            return child.policy_logit + scale * value

        candidates.sort(key=score, reverse=True)

    def _decide_action_child(
        self,
        root,
        candidates,
        select_action_by_count=None,
        select_action_by_softmax_count=None,
        temperature=None,
        value_threshold=None,
    ):
        if select_action_by_count is None:
            select_action_by_count = self.config.select_action_by_count
        if select_action_by_softmax_count is None:
            select_action_by_softmax_count = self.config.select_action_by_softmax_count
        if select_action_by_count == select_action_by_softmax_count:
            raise ValueError("exactly one root action selection mode must be enabled")

        if select_action_by_count:
            ranked_candidates = list(candidates) if candidates else list(root.children.values())
            self._sort_candidates_by_score(root, ranked_candidates)
            return ranked_candidates[0]

        if temperature is None:
            temperature = self.config.select_action_softmax_temperature
        if value_threshold is None:
            value_threshold = self.config.select_action_value_threshold
        return self._select_child_by_softmax_count(root, temperature, value_threshold)

    def _select_child_by_softmax_count(self, root, temperature=1.0, value_threshold=0.1):
        visited_children = [child for child in root.children.values() if child.count > 0]
        if not visited_children:
            return max(root.children.values(), key=lambda child: child.policy_logit)

        best_child = max(visited_children, key=lambda child: child.count)
        if temperature <= 0:
            return best_child

        best_mean = best_child.normalized_mean(self.config)
        eligible = []
        weights = []
        for child in root.children.values():
            if child.count <= 0:
                continue
            mean = child.normalized_mean(self.config)
            if mean < best_mean - value_threshold:
                continue
            weight = child.count ** (1.0 / temperature)
            if weight <= 0:
                continue
            eligible.append(child)
            weights.append(weight)

        if not eligible:
            return best_child

        probs = np.asarray(weights, dtype=np.float64)
        probs /= probs.sum()
        idx = int(np.random.choice(len(eligible), p=probs))
        return eligible[idx]

    def _improved_policy(self, root, player):
        visited_children = [child for child in root.children.values() if child.count > 0]
        pi_sum = sum(child.prior for child in visited_children)
        q_sum = sum(child.prior * child.normalized_mean(self.config) for child in visited_children)

        root_value = root.value
        if player == self.config.value_flipping_player:
            root_value = -root_value

        if pi_sum > 0:
            non_visited_value = (
                root_value + (self.config.num_simulations / pi_sum) * q_sum
            ) / (1 + self.config.num_simulations)
        else:
            non_visited_value = root_value

        max_child_count = max((child.count for child in root.children.values()), default=0)
        scale = (self.config.gumbel_sigma_visit_c + max_child_count) * self.config.gumbel_sigma_scale_c
        logits = np.full(self.config.action_space, -np.inf, dtype=np.float32)
        for child in root.children.values():
            value = non_visited_value if child.count == 0 else child.normalized_mean(self.config)
            logit_without_noise = child.policy_logit - child.policy_noise
            logits[child.action] = logit_without_noise + scale * value
        return _masked_softmax(logits)

    @staticmethod
    def _terminal_value(winner):
        return 1.0 if winner == Player.BLACK else -1.0


class GumbelZeroAgent:
    def __init__(self, config, network, device):
        self.mcts = create_gumbel_mcts(config, network, device)

    def select_move(
        self,
        board,
        player,
        add_noise=False,
        select_action_by_count=None,
        select_action_by_softmax_count=None,
        temperature=None,
        value_threshold=None,
    ):
        action, _ = self.mcts.run(
            board,
            int(player),
            add_noise=add_noise,
            select_action_by_count=select_action_by_count,
            select_action_by_softmax_count=select_action_by_softmax_count,
            temperature=temperature,
            value_threshold=value_threshold,
        )
        return action


def create_gumbel_mcts(config, network, device):
    backend = getattr(config, "mcts_backend", "auto")
    if backend in {"auto", "cython"}:
        try:
            module = importlib.import_module("cgumbel_mcts")
            return module.CGumbelMCTS(config, network, device)
        except Exception:
            if backend == "cython":
                raise
    return GumbelMCTS(config, network, device)
