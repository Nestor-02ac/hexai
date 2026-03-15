"""Gumbel MCTS: sequential halving with Gumbel noise at the root, PUCT interior."""

import math
import numpy as np
import torch

from neural_net import encode_board


def _masked_softmax(logits):
    """Softmax over finite entries, zero for -inf."""
    finite_mask = logits > -np.inf
    if not finite_mask.any():
        return np.zeros_like(logits)
    max_val = logits[finite_mask].max()
    exp = np.where(finite_mask, np.exp(logits - max_val), 0.0)
    total = exp.sum()
    return exp / total if total > 0 else exp


class Node:
    __slots__ = ['prior', 'visit_count', 'value_sum', 'children', 'to_play']

    def __init__(self, prior=0.0, to_play=0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # action -> Node
        self.to_play = to_play

    @property
    def expanded(self):
        return len(self.children) > 0

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class GumbelMCTS:
    def __init__(self, config, network, device):
        self.config = config
        self.network = network
        self.device = device

    def run(self, board, player):
        """
        Run Gumbel MCTS from the given position.
        Returns (action, improved_policy) where improved_policy is the training target.
        """
        root = Node(to_play=player)
        root_value = self._expand(root, board, player)

        legal_actions = list(root.children.keys())
        n_legal = len(legal_actions)

        if n_legal == 0:
            policy = np.zeros(self.config.action_space, dtype=np.float32)
            return 0, policy

        if n_legal == 1:
            policy = np.zeros(self.config.action_space, dtype=np.float32)
            policy[legal_actions[0]] = 1.0
            return legal_actions[0], policy

        # Sample Gumbel noise and compute log-priors
        gumbels = {a: np.random.gumbel() for a in legal_actions}
        log_pi = {a: math.log(root.children[a].prior + 1e-8) for a in legal_actions}

        # Top-m candidates by g(a) + log pi(a)
        m = min(self.config.max_considered_actions, n_legal)
        init_scores = {a: gumbels[a] + log_pi[a] for a in legal_actions}
        candidates = sorted(legal_actions, key=lambda a: init_scores[a], reverse=True)[:m]

        # Sequential halving
        N = self.config.num_simulations
        num_phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
        sims_used = 0

        for phase in range(num_phases):
            k = len(candidates)
            if k <= 1:
                break
            sims_per = max(1, N // (num_phases * k))

            for action in candidates:
                for _ in range(sims_per):
                    if sims_used >= N:
                        break
                    self._simulate(root, action, board, player)
                    sims_used += 1
                if sims_used >= N:
                    break

            # Score and halve
            q_bar = self._completed_q(root, root_value)
            sigma = self._sigma(root, q_bar)
            scored = [(gumbels[a] + log_pi[a] + sigma[a], a) for a in candidates]
            scored.sort(reverse=True)
            candidates = [a for _, a in scored[:max(1, k // 2)]]

        # Spend remaining budget on survivors
        remaining = N - sims_used
        if remaining > 0 and candidates:
            per = max(1, remaining // len(candidates))
            for action in candidates:
                for _ in range(per):
                    if sims_used >= N:
                        break
                    self._simulate(root, action, board, player)
                    sims_used += 1

        # Final action
        q_bar = self._completed_q(root, root_value)
        sigma = self._sigma(root, q_bar)
        best_action = max(legal_actions, key=lambda a: gumbels[a] + log_pi[a] + sigma[a])

        # Improved policy (training target)
        improved = self._improved_policy(log_pi, sigma)

        return best_action, improved

    # -- Batched expansion interface (used by parallel self-play) --

    def prepare_expand(self, node, board, to_play):
        """Encode the board for NN evaluation. Returns (state_tensor, legal_actions)."""
        state = encode_board(board, to_play)
        legal_actions = board.get_empty_cells()
        return state, legal_actions

    def finish_expand(self, node, to_play, policy_logits_np, value, legal_actions):
        """Create children from NN output. Returns value."""
        mask = np.full(self.config.action_space, -np.inf, dtype=np.float32)
        for a in legal_actions:
            mask[a] = policy_logits_np[a]
        priors = _masked_softmax(mask)
        for a in legal_actions:
            node.children[a] = Node(prior=priors[a], to_play=3 - to_play)
        return value

    # -- Single-sample expansion (used by sequential run()) --

    def _expand(self, node, board, to_play):
        """Evaluate leaf with neural net, create children. Returns value."""
        state = encode_board(board, to_play).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.network(state)
        policy_logits = policy_logits.squeeze(0).cpu().numpy()
        value = value.item()
        legal_actions = board.get_empty_cells()
        return self.finish_expand(node, to_play, policy_logits, value, legal_actions)

    def _simulate(self, root, root_action, board, player):
        """One MCTS simulation forced through root_action."""
        scratch = board.clone()
        path = []

        scratch.play(root_action, player)
        path.append((root, root_action))

        if scratch.check_win(player):
            self._backpropagate(path, 1.0)
            return

        current = 3 - player
        node = root.children[root_action]

        while node.expanded:
            action, child = self._select_puct(node)
            scratch.play(action, current)
            path.append((node, action))

            if scratch.check_win(current):
                value = 1.0 if current == player else -1.0
                self._backpropagate(path, value)
                return

            current = 3 - current
            node = child

        # Expand leaf
        value = self._expand(node, scratch, current)
        if current != player:
            value = -value
        self._backpropagate(path, value)

    def simulate_until_leaf(self, root, root_action, board, player):
        """Like _simulate but returns pending leaf info instead of calling NN.
        Returns None if the simulation ended in a terminal state,
        or (node, scratch_board, current_player, path) for the leaf to expand."""
        scratch = board.clone()
        path = []

        scratch.play(root_action, player)
        path.append((root, root_action))

        if scratch.check_win(player):
            self._backpropagate(path, 1.0)
            return None

        current = 3 - player
        node = root.children[root_action]

        while node.expanded:
            action, child = self._select_puct(node)
            scratch.play(action, current)
            path.append((node, action))

            if scratch.check_win(current):
                value = 1.0 if current == player else -1.0
                self._backpropagate(path, value)
                return None

            current = 3 - current
            node = child

        return node, scratch, current, path, player

    def finish_simulate(self, leaf_info, policy_logits_np, value):
        """Finish a simulation after batched NN eval."""
        node, _scratch, current, path, root_player = leaf_info
        legal = leaf_info[1].get_empty_cells()
        self.finish_expand(node, current, policy_logits_np, value, legal)
        if current != root_player:
            value = -value
        self._backpropagate(path, value)

    def _select_puct(self, node):
        """PUCT selection for interior nodes."""
        total = sum(c.visit_count for c in node.children.values())
        c_puct = math.log((total + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
        sqrt_total = math.sqrt(total)

        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            q = child.q_value
            exploration = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + exploration
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _backpropagate(self, path, value_from_root_player):
        """Backprop value through the path, flipping sign at each level."""
        value = value_from_root_player
        for parent, action in path:
            child = parent.children[action]
            child.visit_count += 1
            child.value_sum += value
            value = -value

    def _completed_q(self, node, node_value):
        """Q for visited children, v_mix for unvisited."""
        children = node.children
        total_visits = sum(c.visit_count for c in children.values())

        if total_visits == 0:
            v_mix = node_value
        else:
            visited_sum = sum(c.value_sum for c in children.values() if c.visit_count > 0)
            v_mix = (node_value + visited_sum) / (1 + total_visits)

        return {
            a: (child.q_value if child.visit_count > 0 else v_mix)
            for a, child in children.items()
        }

    def _sigma(self, node, q_bar):
        """Scale completed Q-values for mixing with log-priors."""
        if self.config.value_scale <= 0:
            return {a: 0.0 for a in q_bar}

        max_visit = max((c.visit_count for c in node.children.values()), default=0)
        scale = self.config.maxvisit_init / (self.config.maxvisit_init + max_visit) * self.config.value_scale

        values = list(q_bar.values())
        q_min, q_max = min(values), max(values)
        q_range = q_max - q_min

        result = {}
        for a, q in q_bar.items():
            normalized = (q - q_min) / q_range if q_range > 0 else 0.0
            result[a] = scale * normalized
        return result

    def _improved_policy(self, log_pi, sigma):
        """Training target: softmax(log_pi + sigma(q_bar))."""
        logits = np.full(self.config.action_space, -np.inf, dtype=np.float32)
        for a in log_pi:
            logits[a] = log_pi[a] + sigma[a]
        return _masked_softmax(logits)


class GumbelZeroAgent:
    """Drop-in agent compatible with play_game(). Implements select_move(board, player) -> int."""

    def __init__(self, config, network, device):
        self.mcts = GumbelMCTS(config, network, device)

    def select_move(self, board, player):
        action, _ = self.mcts.run(board, int(player))
        return action
