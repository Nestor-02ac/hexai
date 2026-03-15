"""Benchmark Gumbel AlphaZero against random and classical MCTS."""

import sys
import os
import math
import random as py_random
import time

import numpy as np
import torch

from hex_board import Player
from neural_net import encode_board
from mcts import GumbelMCTS, Node

# Add HexClassic to path for classical MCTS imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'HexClassic'))


def _make_board(board_size):
    try:
        from chex_board import CHexBoard
        return CHexBoard(board_size)
    except ImportError:
        from hex_board import HexBoard
        return HexBoard(board_size)


def _batched_nn_eval(network, device, state_tensors):
    batch = torch.stack(state_tensors).to(device)
    with torch.no_grad():
        policy_logits, values = network(batch)
    return policy_logits.cpu().numpy(), values.cpu().numpy()


# -- Batched eval game runner --

def _play_eval_games_batched(config, network, device, num_games, opponent_factory,
                              gumbel_is_black_fn):
    """Run eval games in parallel with batched NN for the Gumbel agent.

    gumbel_is_black_fn(game_idx) -> bool: whether the Gumbel agent plays BLACK.
    opponent_factory() -> agent with select_move(board, player).
    """
    network.eval()
    board_size = config.board_size

    boards = [_make_board(board_size) for _ in range(num_games)]
    gumbel_is_black = [gumbel_is_black_fn(i) for i in range(num_games)]
    opponents = [opponent_factory() for _ in range(num_games)]
    currents = [1] * num_games
    finished = [False] * num_games
    winners = [0] * num_games

    # MCTS state (only used on gumbel's turn)
    mcts_engines = [GumbelMCTS(config, network, device) for _ in range(num_games)]
    roots = [None] * num_games
    root_values = [None] * num_games
    gumbels = [None] * num_games
    log_pis = [None] * num_games
    candidates_list = [None] * num_games
    legal_list = [None] * num_games
    sims_used = [0] * num_games
    sim_budgets = [0] * num_games
    phase_plans = [None] * num_games
    phase_idx = [0] * num_games
    cand_idx = [0] * num_games
    sims_this_cand = [0] * num_games
    needs_root = [False] * num_games
    in_mcts = [False] * num_games  # True when gumbel is mid-search

    def _is_gumbel_turn(i):
        if gumbel_is_black[i]:
            return currents[i] == 1
        return currents[i] == 2

    def _init_mcts(i):
        roots[i] = Node(to_play=currents[i])
        needs_root[i] = True
        sims_used[i] = 0
        in_mcts[i] = True

    def _setup_halving(i):
        legal = list(roots[i].children.keys())
        legal_list[i] = legal
        if len(legal) <= 1:
            phase_plans[i] = []
            return
        gumbels[i] = {a: np.random.gumbel() for a in legal}
        log_pis[i] = {a: math.log(roots[i].children[a].prior + 1e-8) for a in legal}
        m = min(config.max_considered_actions, len(legal))
        init_scores = {a: gumbels[i][a] + log_pis[i][a] for a in legal}
        cands = sorted(legal, key=lambda a: init_scores[a], reverse=True)[:m]
        N = config.num_simulations
        sim_budgets[i] = N
        num_phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
        plan = []
        remaining_cands = list(cands)
        for _ in range(num_phases):
            k = len(remaining_cands)
            if k <= 1:
                break
            spc = max(1, N // (num_phases * k))
            plan.append((list(remaining_cands), spc))
            remaining_cands = remaining_cands[:max(1, k // 2)]
        phase_plans[i] = plan
        candidates_list[i] = cands
        phase_idx[i] = 0
        cand_idx[i] = 0
        sims_this_cand[i] = 0

    def _current_action(i):
        plan = phase_plans[i]
        if not plan:
            return None
        if phase_idx[i] >= len(plan):
            cands = candidates_list[i]
            if not cands or cand_idx[i] >= len(cands):
                return None
            return cands[cand_idx[i]]
        cands, _ = plan[phase_idx[i]]
        if cand_idx[i] >= len(cands):
            return None
        return cands[cand_idx[i]]

    def _advance_after_sim(i):
        sims_used[i] += 1
        sims_this_cand[i] += 1
        plan = phase_plans[i]
        if not plan and phase_idx[i] == 0:
            return
        if phase_idx[i] < len(plan):
            cands, spc = plan[phase_idx[i]]
            if sims_this_cand[i] >= spc:
                sims_this_cand[i] = 0
                cand_idx[i] += 1
                if cand_idx[i] >= len(cands):
                    q_bar = mcts_engines[i]._completed_q(roots[i], root_values[i])
                    sigma = mcts_engines[i]._sigma(roots[i], q_bar)
                    scored = [(gumbels[i][a] + log_pis[i][a] + sigma[a], a) for a in cands]
                    scored.sort(reverse=True)
                    new_cands = [a for _, a in scored[:max(1, len(cands) // 2)]]
                    candidates_list[i] = new_cands
                    phase_idx[i] += 1
                    cand_idx[i] = 0
                    if phase_idx[i] < len(plan):
                        _, spc_next = plan[phase_idx[i]]
                        plan[phase_idx[i]] = (new_cands, spc_next)
        else:
            cands = candidates_list[i]
            remaining = sim_budgets[i] - sims_used[i]
            per = max(1, (remaining + len(cands)) // len(cands)) if cands else 0
            if sims_this_cand[i] >= per:
                sims_this_cand[i] = 0
                cand_idx[i] += 1

    def _move_ready(i):
        if not phase_plans[i]:
            return True
        if sims_used[i] >= sim_budgets[i]:
            return True
        if _current_action(i) is None:
            return True
        return False

    def _commit_gumbel_move(i):
        legal = legal_list[i]
        if len(legal) <= 1:
            action = legal[0] if legal else 0
        else:
            q_bar = mcts_engines[i]._completed_q(roots[i], root_values[i])
            sigma = mcts_engines[i]._sigma(roots[i], q_bar)
            action = max(legal, key=lambda a: gumbels[i][a] + log_pis[i][a] + sigma[a])
        _play_move(i, action)

    def _play_move(i, action):
        boards[i].play(action, currents[i])
        in_mcts[i] = False
        if boards[i].check_win(currents[i]):
            winners[i] = currents[i]
            finished[i] = True
        else:
            currents[i] = 3 - currents[i]

    # Main loop
    while not all(finished):
        # First: handle opponent turns instantly
        for i in range(num_games):
            if finished[i] or in_mcts[i]:
                continue
            if not _is_gumbel_turn(i):
                move = opponents[i].select_move(boards[i], currents[i])
                _play_move(i, move)
            elif not in_mcts[i]:
                _init_mcts(i)

        # Collect NN eval requests from games in MCTS
        eval_requests = []
        for i in range(num_games):
            if finished[i] or not in_mcts[i]:
                continue

            if needs_root[i]:
                state, legal = mcts_engines[i].prepare_expand(roots[i], boards[i], currents[i])
                eval_requests.append((i, state, legal, 'root', None))
                continue

            if _move_ready(i):
                _commit_gumbel_move(i)
                continue

            action = _current_action(i)
            if action is None:
                _commit_gumbel_move(i)
                continue

            leaf_info = mcts_engines[i].simulate_until_leaf(roots[i], action, boards[i], currents[i])
            if leaf_info is None:
                _advance_after_sim(i)
                if _move_ready(i):
                    _commit_gumbel_move(i)
            else:
                node, scratch, current, path, root_player = leaf_info
                state, legal = mcts_engines[i].prepare_expand(node, scratch, current)
                eval_requests.append((i, state, legal, 'leaf', leaf_info))

        if not eval_requests:
            continue

        states = [req[1] for req in eval_requests]
        policy_batch, value_batch = _batched_nn_eval(network, device, states)

        for j, (i, _state, legal, rtype, extra) in enumerate(eval_requests):
            pol = policy_batch[j]
            val = value_batch[j].item()
            if rtype == 'root':
                root_values[i] = mcts_engines[i].finish_expand(roots[i], currents[i], pol, val, legal)
                needs_root[i] = False
                _setup_halving(i)
                if _move_ready(i):
                    _commit_gumbel_move(i)
            else:
                node, scratch, current, path, root_player = extra
                value = mcts_engines[i].finish_expand(node, current, pol, val, legal)
                if current != root_player:
                    value = -value
                mcts_engines[i]._backpropagate(path, value)
                _advance_after_sim(i)
                if _move_ready(i):
                    _commit_gumbel_move(i)

    return winners, gumbel_is_black


# -- Public API --

class RandomAgent:
    def select_move(self, board, player):
        return py_random.choice(board.get_empty_cells())


def evaluate_vs_random(config, network, device, num_games=None):
    if num_games is None:
        num_games = config.eval_games

    winners, gumbel_is_black = _play_eval_games_batched(
        config, network, device, num_games,
        opponent_factory=RandomAgent,
        gumbel_is_black_fn=lambda i: i % 2 == 0,
    )
    wins = sum(
        1 for i in range(num_games)
        if (gumbel_is_black[i] and winners[i] == 1) or
           (not gumbel_is_black[i] and winners[i] == 2)
    )
    return 100.0 * wins / num_games


def evaluate_vs_classical_mcts(config, network, device, mcts_sims=None, num_games=None):
    if num_games is None:
        num_games = config.eval_games
    if mcts_sims is None:
        mcts_sims = config.eval_mcts_simulations

    try:
        from cmcts_hex import CMCTSHex
        from mcts_hex import SimulationType
        ClassicMCTS = CMCTSHex
    except ImportError:
        try:
            from mcts_hex import MCTSHex, SimulationType
            ClassicMCTS = MCTSHex
        except ImportError:
            print("  Classical MCTS not available (HexClassic not in path), skipping.")
            return None

    def make_opponent():
        return ClassicMCTS(
            board_size=config.board_size,
            c_uct=0.0,
            rave_bias=0.00025,
            use_rave=True,
            simulation_type=SimulationType.BRIDGES,
            num_simulations=mcts_sims,
        )

    winners, gumbel_is_black = _play_eval_games_batched(
        config, network, device, num_games,
        opponent_factory=make_opponent,
        gumbel_is_black_fn=lambda i: i % 2 == 0,
    )
    wins = sum(
        1 for i in range(num_games)
        if (gumbel_is_black[i] and winners[i] == 1) or
           (not gumbel_is_black[i] and winners[i] == 2)
    )
    return 100.0 * wins / num_games


def run_evaluation(config, network, device, iteration):
    network.eval()
    print(f"\n  Evaluation at iteration {iteration}")

    t0 = time.time()
    wr_random = evaluate_vs_random(config, network, device)
    print(f"  vs Random: {wr_random:.1f}% ({time.time() - t0:.1f}s)")

    t0 = time.time()
    wr_mcts = evaluate_vs_classical_mcts(config, network, device)
    if wr_mcts is not None:
        print(f"  vs MCTS ({config.eval_mcts_simulations} sims): {wr_mcts:.1f}% ({time.time() - t0:.1f}s)")

    return {'vs_random': wr_random, 'vs_mcts': wr_mcts}
