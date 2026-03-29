# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython-accelerated Gumbel AlphaZero search for Hex."""

from libc.math cimport exp as cexp, log as clog, pow as cpow, sqrt

import math
import numpy as np
import torch
cimport numpy as cnp

from chex_board cimport CHexBoard, encode_board_tensor_c
from hex_board import Player


def _masked_softmax(logits):
    finite_mask = np.isfinite(logits)
    if not finite_mask.any():
        return np.zeros_like(logits, dtype=np.float32)
    max_val = logits[finite_mask].max()
    exp = np.zeros_like(logits, dtype=np.float32)
    exp[finite_mask] = np.exp(logits[finite_mask] - max_val)
    total = exp.sum()
    return exp / total if total > 0 else exp


cdef CHexBoard _python_board_to_cboard(object board_obj):
    cdef int size = int(board_obj.size)
    cdef int n = size * size
    cdef int idx
    cdef int cell
    cdef CHexBoard board = CHexBoard(size)
    for idx in range(n):
        cell = int(board_obj.get_cell(idx))
        if cell != 0:
            board.play_unchecked(idx, cell)
    return board


cdef inline CHexBoard _to_cboard(object board_obj):
    if isinstance(board_obj, CHexBoard):
        return <CHexBoard>board_obj
    return _python_board_to_cboard(board_obj)


cdef class CNode:
    cdef public int action
    cdef public int action_player
    cdef public double prior
    cdef public double policy_logit
    cdef public double policy_noise
    cdef public double value
    cdef public double reward
    cdef public double mean
    cdef public int count
    cdef public list children

    def __init__(self, int action=-1, int action_player=0, double prior=0.0, double policy_logit=0.0):
        self.action = action
        self.action_player = action_player
        self.prior = prior
        self.policy_logit = policy_logit
        self.policy_noise = 0.0
        self.value = 0.0
        self.reward = 0.0
        self.mean = 0.0
        self.count = 0
        self.children = []

    cdef inline bint expanded(self):
        return len(self.children) != 0

    cdef inline void add(self, double value):
        self.count += 1
        self.mean += (value - self.mean) / self.count

    cdef inline double normalized_mean(self, object config):
        cdef double value = self.reward + config.reward_discount * self.mean
        if self.action_player == config.value_flipping_player:
            value = -value
        return value

    cdef inline double puct_score(self, int total_simulation, object config, double init_q_value):
        cdef double puct_bias = config.pb_c_init + clog((1 + total_simulation + config.pb_c_base) / config.pb_c_base)
        cdef double value_u = (puct_bias * self.prior * sqrt(max(1, total_simulation))) / (1 + self.count)
        cdef double value_q = init_q_value if self.count == 0 else self.normalized_mean(config)
        return value_u + value_q


cdef class CRootSearchState:
    cdef public CNode root
    cdef public int player
    cdef public double root_value
    cdef public list legal_actions
    cdef public list candidates
    cdef public int sample_size
    cdef public int simulation_budget

    def __init__(self, CNode root, int player):
        self.root = root
        self.player = player
        self.root_value = 0.0
        self.legal_actions = []
        self.candidates = []
        self.sample_size = 0
        self.simulation_budget = 0


cdef class CLeafRequest:
    cdef public CRootSearchState search
    cdef public CNode node
    cdef public CHexBoard board
    cdef public int to_play
    cdef public list path

    def __init__(self, CRootSearchState search, CNode node, CHexBoard board, int to_play, list path):
        self.search = search
        self.node = node
        self.board = board
        self.to_play = to_play
        self.path = path


cdef class CGumbelMCTS:
    cdef public object config
    cdef public object network
    cdef public object device
    cdef int action_space
    cdef int num_simulations
    cdef int gumbel_sample_size
    cdef int value_flipping_player
    cdef double pb_c_base
    cdef double pb_c_init
    cdef double reward_discount
    cdef double gumbel_sigma_visit_c
    cdef double gumbel_sigma_scale_c
    cdef bint default_select_action_by_count
    cdef bint default_select_action_by_softmax_count
    cdef double default_temperature
    cdef double default_value_threshold

    def __init__(self, config, network, device):
        self.config = config
        self.network = network
        self.device = device
        self.action_space = config.action_space
        self.num_simulations = config.num_simulations
        self.gumbel_sample_size = config.gumbel_sample_size
        self.value_flipping_player = config.value_flipping_player
        self.pb_c_base = config.pb_c_base
        self.pb_c_init = config.pb_c_init
        self.reward_discount = config.reward_discount
        self.gumbel_sigma_visit_c = config.gumbel_sigma_visit_c
        self.gumbel_sigma_scale_c = config.gumbel_sigma_scale_c
        self.default_select_action_by_count = config.select_action_by_count
        self.default_select_action_by_softmax_count = config.select_action_by_softmax_count
        self.default_temperature = config.select_action_softmax_temperature
        self.default_value_threshold = config.select_action_value_threshold

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
        cdef CRootSearchState search = self.new_search(player)
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

    def new_search(self, int player):
        return CRootSearchState(CNode(action=-1, action_player=3 - player), player)

    def prepare_expand(self, CNode node, board, int to_play):
        cdef CHexBoard cboard = _to_cboard(board)
        return encode_board_tensor_c(cboard, to_play), cboard.get_empty_cells()

    def finish_root(self, CRootSearchState search, policy_logits_np, double value, legal_actions, bint add_noise=False):
        search.root_value = self._expand_from_logits(
            search.root,
            search.player,
            policy_logits_np,
            value,
            legal_actions,
            add_noise=add_noise,
        )
        self._backup([search.root], search.root_value)
        search.legal_actions = list(legal_actions)
        self._initialize_candidates(search)
        return search.root_value

    def simulate_until_leaf(self, CRootSearchState search, board):
        cdef CNode candidate
        cdef CHexBoard scratch
        cdef list path
        cdef int current
        cdef CNode node

        if self.search_complete(search):
            return None

        candidate = self._select_root_candidate(search)
        if candidate is None:
            return None

        scratch = _to_cboard(board).clone()
        path = [search.root, candidate]

        scratch.play_unchecked(candidate.action, search.player)
        if scratch.check_win(search.player):
            self._backup(path, self._terminal_value(search.player))
            self._advance_search(search)
            return None

        current = 3 - search.player
        node = candidate
        while node.expanded():
            node = self._select_child_by_puct(node)
            scratch.play_unchecked(node.action, current)
            path.append(node)
            if scratch.check_win(current):
                self._backup(path, self._terminal_value(current))
                self._advance_search(search)
                return None
            current = 3 - current

        return CLeafRequest(search, node, scratch, current, path)

    def finish_leaf(self, CLeafRequest leaf_request, policy_logits_np, double value, legal_actions):
        cdef double expanded_value = self._expand_from_logits(
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

    def search_complete(self, CRootSearchState search):
        if not search.legal_actions:
            return True
        if len(search.legal_actions) <= 1:
            return True
        return search.root.count >= self.num_simulations + 1

    cdef inline double _normalized_mean(self, CNode node):
        cdef double value = node.reward + self.reward_discount * node.mean
        if node.action_player == self.value_flipping_player:
            value = -value
        return value

    cdef inline double _puct_score(self, CNode child, int total_simulation, double init_q_value):
        cdef double puct_bias = self.pb_c_init + clog((1 + total_simulation + self.pb_c_base) / self.pb_c_base)
        cdef double value_u = (puct_bias * child.prior * sqrt(max(1, total_simulation))) / (1 + child.count)
        cdef double value_q = init_q_value if child.count == 0 else self._normalized_mean(child)
        return value_u + value_q

    def finalize_search(
        self,
        CRootSearchState search,
        select_action_by_count=None,
        select_action_by_softmax_count=None,
        temperature=None,
        value_threshold=None,
    ):
        cdef object policy = np.zeros(self.action_space, dtype=np.float32)
        cdef CNode child
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

    cdef double _expand_from_logits(self, CNode node, int to_play, policy_logits_np, double value, legal_actions, bint add_noise=False):
        cdef cnp.ndarray[cnp.float32_t, ndim=1] logits = np.asarray(policy_logits_np, dtype=np.float32).reshape(-1)
        cdef list children
        cdef object noise
        cdef Py_ssize_t idx
        cdef Py_ssize_t num_legal = len(legal_actions)
        cdef int action
        cdef double logit
        cdef double max_logit = -1e300
        cdef double prior_value
        cdef double prior_sum = 0.0
        cdef CNode child

        node.value = value
        if num_legal == 0:
            node.children = []
            return node.value

        children = [None] * num_legal
        for idx in range(num_legal):
            action = <int>legal_actions[idx]
            logit = <double>logits[action]
            if logit > max_logit:
                max_logit = logit

        for idx in range(num_legal):
            action = <int>legal_actions[idx]
            logit = <double>logits[action]
            prior_value = cexp(logit - max_logit)
            prior_sum += prior_value
            children[idx] = CNode(
                action=action,
                action_player=to_play,
                prior=prior_value,
                policy_logit=logit,
            )

        if prior_sum > 0.0:
            for idx in range(num_legal):
                child = <CNode>children[idx]
                child.prior /= prior_sum
        else:
            prior_value = 1.0 / num_legal
            for idx in range(num_legal):
                child = <CNode>children[idx]
                child.prior = prior_value
        node.children = children

        if add_noise and node.children:
            noise = np.random.gumbel(size=num_legal)
            for idx in range(num_legal):
                child = <CNode>node.children[idx]
                child.policy_noise = float(noise[idx])
                child.policy_logit += float(noise[idx])

        return node.value

    cdef CNode _select_root_candidate(self, CRootSearchState search):
        cdef CNode best_child = None
        cdef CNode child
        cdef int best_count = 2147483647
        cdef double best_logit = -1e300
        if not search.candidates:
            return None
        for child in search.candidates:
            if child.count < best_count or (child.count == best_count and child.policy_logit > best_logit):
                best_count = child.count
                best_logit = child.policy_logit
                best_child = child
        return best_child

    cdef CNode _select_child_by_puct(self, CNode node):
        cdef int total_simulation = max(0, node.count - 1)
        cdef double init_q_value = self._calculate_init_q_value(node)
        cdef CNode best_child = None
        cdef CNode child
        cdef double best_score = -1e300
        cdef double best_prior = -1e300
        cdef double score
        for child in node.children:
            score = self._puct_score(child, total_simulation, init_q_value)
            if score > best_score or (score == best_score and child.prior > best_prior):
                best_score = score
                best_prior = child.prior
                best_child = child
        return best_child

    cdef double _calculate_init_q_value(self, CNode node):
        cdef double total = 0.0
        cdef int count = 0
        cdef CNode child
        for child in node.children:
            if child.count > 0:
                total += self._normalized_mean(child)
                count += 1
        if count == 0:
            return -1.0
        return (total - 1.0) / (count + 1.0)

    cdef void _backup(self, list path, double value):
        cdef double updated_value = value
        cdef CNode node
        (<CNode>path[len(path) - 1]).value = value
        for node in reversed(path):
            node.add(updated_value)
            updated_value = node.reward + self.reward_discount * updated_value

    cdef void _sort_nodes_by_logit(self, list nodes):
        cdef Py_ssize_t i
        cdef Py_ssize_t j
        cdef CNode key_node
        for i in range(1, len(nodes)):
            key_node = <CNode>nodes[i]
            j = i - 1
            while j >= 0 and (<CNode>nodes[j]).policy_logit < key_node.policy_logit:
                nodes[j + 1] = nodes[j]
                j -= 1
            nodes[j + 1] = key_node

    def _initialize_candidates(self, CRootSearchState search):
        search.candidates = list(search.root.children)
        self._sort_nodes_by_logit(search.candidates)
        if len(search.candidates) > self.gumbel_sample_size:
            del search.candidates[self.gumbel_sample_size:]
        search.sample_size = self.gumbel_sample_size
        if len(search.candidates) <= 1:
            search.simulation_budget = self.num_simulations
        else:
            search.simulation_budget = self._initial_budget(search.sample_size)

    def _advance_search(self, CRootSearchState search):
        cdef CNode node
        cdef int next_budget
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
            del search.candidates[search.sample_size:]
        if search.candidates:
            search.simulation_budget = (<CNode>search.candidates[0]).count + next_budget

    def _initial_budget(self, int sample_size):
        cdef double log_sample_size
        cdef double denom
        if sample_size <= 1:
            return self.num_simulations
        log_sample_size = math.log2(self.gumbel_sample_size)
        denom = log_sample_size * sample_size
        if denom <= 0:
            return self.num_simulations
        return max(1, int(math.floor(self.num_simulations / denom)))

    def _next_phase_budget(self, int sample_size):
        cdef double log_sample_size
        cdef double denom
        if sample_size <= 1:
            return 0
        log_sample_size = math.log2(self.gumbel_sample_size)
        denom = log_sample_size * sample_size / 2.0
        if denom <= 0:
            return 0
        return int(math.floor(self.num_simulations / denom))

    def _candidate_sigma_score(self, CNode child, int max_child_count):
        cdef double value
        cdef double scale
        if child.count == 0:
            return -1e300
        value = self._normalized_mean(child)
        scale = (self.gumbel_sigma_visit_c + max_child_count) * self.gumbel_sigma_scale_c
        return child.policy_logit + scale * value

    def _sort_candidates_by_score(self, CNode root, list candidates):
        cdef Py_ssize_t i
        cdef Py_ssize_t j
        cdef int max_child_count = 0
        cdef int child_count
        cdef CNode child
        cdef CNode key_node

        for child in root.children:
            child_count = child.count
            if child_count > max_child_count:
                max_child_count = child_count

        for i in range(1, len(candidates)):
            key_node = <CNode>candidates[i]
            j = i - 1
            while j >= 0 and self._candidate_sigma_score(<CNode>candidates[j], max_child_count) < self._candidate_sigma_score(key_node, max_child_count):
                candidates[j + 1] = candidates[j]
                j -= 1
            candidates[j + 1] = key_node

    def _decide_action_child(
        self,
        CNode root,
        list candidates,
        select_action_by_count=None,
        select_action_by_softmax_count=None,
        temperature=None,
        value_threshold=None,
    ):
        cdef list ranked_candidates
        if select_action_by_count is None:
            select_action_by_count = self.default_select_action_by_count
        if select_action_by_softmax_count is None:
            select_action_by_softmax_count = self.default_select_action_by_softmax_count
        if select_action_by_count == select_action_by_softmax_count:
            raise ValueError("exactly one root action selection mode must be enabled")

        if select_action_by_count:
            ranked_candidates = list(candidates) if candidates else list(root.children)
            self._sort_candidates_by_score(root, ranked_candidates)
            return ranked_candidates[0]

        if temperature is None:
            temperature = self.default_temperature
        if value_threshold is None:
            value_threshold = self.default_value_threshold
        return self._select_child_by_softmax_count(root, temperature, value_threshold)

    def _select_child_by_softmax_count(self, CNode root, double temperature=1.0, double value_threshold=0.1):
        cdef CNode best_child = None
        cdef CNode fallback_child = None
        cdef double best_mean
        cdef CNode child
        cdef double mean
        cdef double weight
        cdef double total_weight = 0.0
        cdef double draw
        cdef double cumulative = 0.0
        cdef int best_count = -1
        cdef double best_logit = -1e300
        cdef list eligible = []
        cdef list weights = []
        cdef Py_ssize_t idx

        for child in root.children:
            if child.policy_logit > best_logit:
                best_logit = child.policy_logit
                fallback_child = child
            if child.count > best_count:
                best_count = child.count
                best_child = child

        if best_count <= 0:
            return fallback_child
        if temperature <= 0:
            return best_child

        best_mean = self._normalized_mean(best_child)
        for child in root.children:
            if child.count <= 0:
                continue
            mean = self._normalized_mean(child)
            if mean < best_mean - value_threshold:
                continue
            weight = cpow(child.count, 1.0 / temperature)
            if weight <= 0:
                continue
            eligible.append(child)
            weights.append(weight)
            total_weight += weight

        if not eligible:
            return best_child

        draw = float(np.random.random()) * total_weight
        for idx in range(len(eligible)):
            cumulative += <double>weights[idx]
            if draw <= cumulative:
                return <CNode>eligible[idx]
        return <CNode>eligible[len(eligible) - 1]

    def _improved_policy(self, CNode root, int player):
        cdef double pi_sum = 0.0
        cdef double q_sum = 0.0
        cdef double root_value = root.value
        cdef double non_visited_value
        cdef int max_child_count = 0
        cdef double scale
        cdef object logits = np.full(self.action_space, -np.inf, dtype=np.float32)
        cdef CNode child
        cdef double value
        cdef double logit_without_noise

        for child in root.children:
            if child.count <= 0:
                continue
            pi_sum += child.prior
            q_sum += child.prior * self._normalized_mean(child)

        if player == self.value_flipping_player:
            root_value = -root_value

        if pi_sum > 0:
            non_visited_value = (
                root_value + (self.num_simulations / pi_sum) * q_sum
            ) / (1 + self.num_simulations)
        else:
            non_visited_value = root_value

        for child in root.children:
            if child.count > max_child_count:
                max_child_count = child.count
        scale = (self.gumbel_sigma_visit_c + max_child_count) * self.gumbel_sigma_scale_c
        for child in root.children:
            value = non_visited_value if child.count == 0 else self._normalized_mean(child)
            logit_without_noise = child.policy_logit - child.policy_noise
            logits[child.action] = logit_without_noise + scale * value
        return _masked_softmax(logits)

    @staticmethod
    def _terminal_value(int winner):
        return 1.0 if winner == Player.BLACK else -1.0
