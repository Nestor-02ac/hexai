"""
Benchmark the Y Gumbel agent against random and classical MCTS.
"""

import os
import random as py_random
import sys
import time

from y_board import Player
from mcts import create_gumbel_mcts
from progress import make_progress

# Optional: if you want to reuse your YClassic folder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "YClassic"))


def _make_board(board_size):
    from y_board import YBoard
    return YBoard(board_size)


class RandomAgent:
    def select_move(self, board, player):
        return py_random.choice(board.get_empty_cells())


def _play_eval_games_batched(config, network, device, num_games, opponent_factory, gumbel_is_black_fn, progress_desc):
    network.eval()
    mcts = create_gumbel_mcts(config, network, device)

    progress = make_progress(
        total=num_games,
        desc=progress_desc,
        unit="game",
        enabled=getattr(config, "show_progress_bars", True),
    )

    boards = [_make_board(config.board_size) for _ in range(num_games)]
    gumbel_is_black = [gumbel_is_black_fn(i) for i in range(num_games)]
    opponents = [opponent_factory() for _ in range(num_games)]

    currents = [Player.BLACK for _ in range(num_games)]
    searches = [None] * num_games
    in_search = [False] * num_games
    finished = [False] * num_games
    winners = [Player.EMPTY for _ in range(num_games)]

    moves_played = [0]

    def _refresh_progress(force=False):
        if force or moves_played[0] == 1 or moves_played[0] % 8 == 0:
            active = sum(0 if f else 1 for f in finished)
            progress.set_postfix(moves=moves_played[0], active=active)

    def _is_gumbel_turn(i):
        return (currents[i] == Player.BLACK) if gumbel_is_black[i] else (currents[i] == Player.WHITE)

    def _start_gumbel_turn(i):
        searches[i] = mcts.new_search(int(currents[i]))
        in_search[i] = True

    def _play_move(i, action):
        boards[i].play(action, currents[i])
        moves_played[0] += 1
        _refresh_progress()

        if boards[i].check_win(currents[i]):
            winners[i] = currents[i]
            finished[i] = True
            progress.update(1)
            _refresh_progress(force=True)
            return

        currents[i] = Player.WHITE if currents[i] == Player.BLACK else Player.BLACK

    def _commit_gumbel_move(i):
        action, _ = mcts.finalize_search(
            searches[i],
            select_action_by_count=config.eval_select_action_by_count,
            select_action_by_softmax_count=config.eval_select_action_by_softmax_count,
            temperature=config.eval_select_action_softmax_temperature,
            value_threshold=config.eval_select_action_value_threshold,
        )
        in_search[i] = False
        _play_move(i, action)

    while not all(finished):

        # play moves
        for i in range(num_games):
            if finished[i] or in_search[i]:
                continue

            if _is_gumbel_turn(i):
                _start_gumbel_turn(i)
            else:
                move = opponents[i].select_move(boards[i], currents[i])
                _play_move(i, move)

        # batched NN evaluation
        eval_requests = []

        for i in range(num_games):
            if finished[i] or not in_search[i]:
                continue

            search = searches[i]

            if search.root.count == 0:
                state, legal = mcts.prepare_expand(search.root, boards[i], int(currents[i]))
                eval_requests.append(("root", i, state, legal, None))
                continue

            if mcts.search_complete(search):
                _commit_gumbel_move(i)
                continue

            leaf = mcts.simulate_until_leaf(search, boards[i])
            if leaf is None:
                if mcts.search_complete(search):
                    _commit_gumbel_move(i)
                continue

            state, legal = mcts.prepare_expand(leaf.node, leaf.board, leaf.to_play)
            eval_requests.append(("leaf", i, state, legal, leaf))

        if not eval_requests:
            continue

        policy_batch, value_batch = mcts.evaluate_states([r[2] for r in eval_requests])

        for idx, (rtype, i, _, legal, extra) in enumerate(eval_requests):
            policy_logits = policy_batch[idx]
            value = float(value_batch[idx])

            if rtype == "root":
                mcts.finish_root(searches[i], policy_logits, value, legal, add_noise=False)
            else:
                mcts.finish_leaf(extra, policy_logits, value, legal)

    _refresh_progress(force=True)
    progress.close()

    return winners, gumbel_is_black


# EVALUATION MODES

def evaluate_vs_random(config, network, device, num_games=None):
    if num_games is None:
        num_games = config.eval_games

    winners, colors = _play_eval_games_batched(
        config,
        network,
        device,
        num_games,
        opponent_factory=RandomAgent,
        gumbel_is_black_fn=lambda i: i % 2 == 0,
        progress_desc="    eval random",
    )

    wins = 0
    for i in range(num_games):
        if (colors[i] and winners[i] == Player.BLACK) or (not colors[i] and winners[i] == Player.WHITE):
            wins += 1

    return 100.0 * wins / num_games


def evaluate_vs_classical_mcts(config, network, device, mcts_sims=None, num_games=None):
    if num_games is None:
        num_games = config.eval_games
    if mcts_sims is None:
        mcts_sims = config.eval_mcts_simulations

    try:
        from mcts import MCTSY

        def make_opponent():
            return MCTSY(
                board_size=config.board_size,
                num_simulations=mcts_sims,
            )

    except ImportError:
        print("  Classical Y MCTS not available, skipping.")
        return None

    winners, colors = _play_eval_games_batched(
        config,
        network,
        device,
        num_games,
        opponent_factory=make_opponent,
        gumbel_is_black_fn=lambda i: i % 2 == 0,
        progress_desc="    eval MCTS",
    )

    wins = 0
    for i in range(num_games):
        if (colors[i] and winners[i] == Player.BLACK) or (not colors[i] and winners[i] == Player.WHITE):
            wins += 1

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

    return {
        "vs_random": wr_random,
        "vs_mcts": wr_mcts,
    }