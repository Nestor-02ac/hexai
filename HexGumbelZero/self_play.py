"""Self-play game generation and replay buffer."""

import math
import numpy as np
import torch

from hex_board import Player
from neural_net import encode_board
from mcts import GumbelMCTS, Node, _masked_softmax


class ReplayBuffer:
    """Circular buffer of (state, policy, value) tuples stored as numpy arrays."""

    def __init__(self, capacity, board_size):
        self.capacity = capacity
        n = board_size * board_size
        self.states = np.zeros((capacity, 3, board_size, board_size), dtype=np.float32)
        self.policies = np.zeros((capacity, n), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.size = 0
        self.index = 0

    def add(self, state, policy, value):
        self.states[self.index] = state
        self.policies[self.index] = policy
        self.values[self.index] = value
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_game(self, game_data):
        for state, policy, value in game_data:
            self.add(state, policy, value)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, size=min(batch_size, self.size), replace=False)
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.policies[indices]),
            torch.from_numpy(self.values[indices]),
        )

    def __len__(self):
        return self.size


def _make_board(board_size):
    """Create a board, preferring Cython if available."""
    try:
        from chex_board import CHexBoard
        return CHexBoard(board_size)
    except ImportError:
        from hex_board import HexBoard
        return HexBoard(board_size)


def _batched_nn_eval(network, device, state_tensors):
    """Run NN on a batch of state tensors. Returns (policy_logits, values) as numpy."""
    batch = torch.stack(state_tensors).to(device)
    with torch.no_grad():
        policy_logits, values = network(batch)
    return policy_logits.cpu().numpy(), values.cpu().numpy()


def generate_self_play_data(config, network, num_games, device):
    """Generate self-play games with batched NN inference across parallel games.

    The key insight: MCTS simulations are sequential within a game (each one
    updates the tree), but we can batch NN evaluations *across* games. So we
    run all games in lockstep: each "tick", every active game runs one simulation
    down to an unexpanded leaf, we batch all those leaf evaluations into one NN
    call, then finish expanding.
    """
    network.eval()
    action_space = config.action_space

    # Per-game state
    boards = [_make_board(config.board_size) for _ in range(num_games)]
    mcts_engines = [GumbelMCTS(config, network, device) for _ in range(num_games)]
    currents = [1] * num_games  # BLACK starts
    trajectories = [[] for _ in range(num_games)]
    finished = [False] * num_games
    winners = [0] * num_games

    # Per-move MCTS state
    roots = [None] * num_games
    root_values = [None] * num_games
    gumbels = [None] * num_games
    log_pis = [None] * num_games
    candidates_list = [None] * num_games
    legal_list = [None] * num_games
    sims_used = [0] * num_games
    sim_budgets = [0] * num_games

    # Sequential halving schedule
    phase_plans = [None] * num_games  # list of (candidate_set, sims_per_cand)
    phase_idx = [0] * num_games
    cand_idx = [0] * num_games
    sims_this_cand = [0] * num_games
    needs_root = [True] * num_games

    def _init_move(i):
        """Prepare a new move's MCTS search."""
        roots[i] = Node(to_play=currents[i])
        needs_root[i] = True
        sims_used[i] = 0

    def _setup_halving(i):
        """After root expansion, set up sequential halving schedule."""
        legal = list(roots[i].children.keys())
        legal_list[i] = legal
        n_legal = len(legal)

        if n_legal <= 1:
            # Only one legal move, skip search
            phase_plans[i] = []
            return

        gumbels[i] = {a: np.random.gumbel() for a in legal}
        log_pis[i] = {a: math.log(roots[i].children[a].prior + 1e-8) for a in legal}

        m = min(config.max_considered_actions, n_legal)
        init_scores = {a: gumbels[i][a] + log_pis[i][a] for a in legal}
        cands = sorted(legal, key=lambda a: init_scores[a], reverse=True)[:m]

        N = config.num_simulations
        sim_budgets[i] = N
        num_phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1

        # Plan phases: list of (candidates, sims_per_candidate)
        plan = []
        budget_used = 0
        remaining_cands = list(cands)
        for ph in range(num_phases):
            k = len(remaining_cands)
            if k <= 1:
                break
            spc = max(1, N // (num_phases * k))
            plan.append((list(remaining_cands), spc))
            budget_used += spc * k
            remaining_cands = remaining_cands[:max(1, k // 2)]  # placeholder, will be updated

        phase_plans[i] = plan
        candidates_list[i] = cands
        phase_idx[i] = 0
        cand_idx[i] = 0
        sims_this_cand[i] = 0

    def _current_action(i):
        """Get the action for the current simulation."""
        plan = phase_plans[i]
        if not plan:
            return None
        if phase_idx[i] >= len(plan):
            # We're in the "remaining budget" phase
            cands = candidates_list[i]
            if not cands or cand_idx[i] >= len(cands):
                return None
            return cands[cand_idx[i]]
        cands, _spc = plan[phase_idx[i]]
        if cand_idx[i] >= len(cands):
            return None
        return cands[cand_idx[i]]

    def _advance_after_sim(i):
        """Advance indices after one simulation completes."""
        sims_used[i] += 1
        sims_this_cand[i] += 1

        plan = phase_plans[i]
        if not plan and phase_idx[i] == 0:
            return  # single-move case

        if phase_idx[i] < len(plan):
            # In sequential halving phase
            cands, spc = plan[phase_idx[i]]
            if sims_this_cand[i] >= spc:
                sims_this_cand[i] = 0
                cand_idx[i] += 1
                if cand_idx[i] >= len(cands):
                    # End of this phase — halve
                    q_bar = mcts_engines[i]._completed_q(roots[i], root_values[i])
                    sigma = mcts_engines[i]._sigma(roots[i], q_bar)
                    scored = [(gumbels[i][a] + log_pis[i][a] + sigma[a], a) for a in cands]
                    scored.sort(reverse=True)
                    new_cands = [a for _, a in scored[:max(1, len(cands) // 2)]]
                    candidates_list[i] = new_cands

                    phase_idx[i] += 1
                    cand_idx[i] = 0

                    if phase_idx[i] < len(plan):
                        # Update the next phase's candidate list
                        old_cands, spc_next = plan[phase_idx[i]]
                        plan[phase_idx[i]] = (new_cands, spc_next)
        else:
            # Remaining budget phase
            cands = candidates_list[i]
            remaining = sim_budgets[i] - sims_used[i]
            per = max(1, (remaining + len(cands)) // len(cands)) if cands else 0
            if sims_this_cand[i] >= per:
                sims_this_cand[i] = 0
                cand_idx[i] += 1

    def _move_ready(i):
        """Check if search is complete for game i."""
        if not phase_plans[i]:
            return True  # single legal move
        if sims_used[i] >= sim_budgets[i]:
            return True
        if _current_action(i) is None:
            return True
        return False

    def _commit_move(i):
        """Select action, record trajectory, play move.
        No temperature sampling — Gumbel noise already provides exploration."""
        legal = legal_list[i]
        if len(legal) <= 1:
            action = legal[0] if legal else 0
            ip = np.zeros(action_space, dtype=np.float32)
            if legal:
                ip[action] = 1.0
        else:
            q_bar = mcts_engines[i]._completed_q(roots[i], root_values[i])
            sigma = mcts_engines[i]._sigma(roots[i], q_bar)
            action = max(legal, key=lambda a: gumbels[i][a] + log_pis[i][a] + sigma[a])
            ip = mcts_engines[i]._improved_policy(log_pis[i], sigma)

        state_np = encode_board(boards[i], currents[i]).numpy()
        trajectories[i].append((state_np, ip, currents[i]))

        boards[i].play(action, currents[i])

        if boards[i].check_win(currents[i]):
            winners[i] = currents[i]
            finished[i] = True
        else:
            currents[i] = 3 - currents[i]
            _init_move(i)

    # Initialize all games
    for i in range(num_games):
        _init_move(i)

    # Main loop
    while not all(finished):
        # 1. Collect NN evaluation requests
        eval_requests = []  # (game_idx, node, board_for_eval, to_play, request_type, extra)

        for i in range(num_games):
            if finished[i]:
                continue

            if needs_root[i]:
                state, legal = mcts_engines[i].prepare_expand(roots[i], boards[i], currents[i])
                eval_requests.append((i, state, legal, 'root', None))
                continue

            if _move_ready(i):
                _commit_move(i)
                continue

            # Run one simulation to a leaf
            action = _current_action(i)
            if action is None:
                _commit_move(i)
                continue

            leaf_info = mcts_engines[i].simulate_until_leaf(roots[i], action, boards[i], currents[i])
            if leaf_info is None:
                # Terminal node
                _advance_after_sim(i)
                if _move_ready(i):
                    _commit_move(i)
            else:
                node, scratch, current, path, root_player = leaf_info
                state, legal = mcts_engines[i].prepare_expand(node, scratch, current)
                eval_requests.append((i, state, legal, 'leaf', leaf_info))

        if not eval_requests:
            continue

        # 2. Batch NN evaluation
        states = [req[1] for req in eval_requests]
        policy_batch, value_batch = _batched_nn_eval(network, device, states)

        # 3. Apply results
        for j, (i, _state, legal, rtype, extra) in enumerate(eval_requests):
            pol = policy_batch[j]
            val = value_batch[j].item()

            if rtype == 'root':
                root_values[i] = mcts_engines[i].finish_expand(roots[i], currents[i], pol, val, legal)
                needs_root[i] = False
                _setup_halving(i)
                if _move_ready(i):
                    _commit_move(i)
            else:
                leaf_info = extra
                node, scratch, current, path, root_player = leaf_info
                value = mcts_engines[i].finish_expand(node, current, pol, val, legal)
                if current != root_player:
                    value = -value
                mcts_engines[i]._backpropagate(path, value)
                _advance_after_sim(i)
                if _move_ready(i):
                    _commit_move(i)

    # Build results
    all_data = []
    for i in range(num_games):
        game_data = []
        for state_np, policy_np, player_at_pos in trajectories[i]:
            value = 1.0 if player_at_pos == winners[i] else -1.0
            game_data.append((state_np, policy_np, value))
        all_data.append(game_data)

    return all_data


def play_self_play_game(config, network, device):
    """Play one self-play game (convenience wrapper)."""
    return generate_self_play_data(config, network, 1, device)[0]
