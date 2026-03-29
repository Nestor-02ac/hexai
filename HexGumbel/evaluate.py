"""Benchmark the Hex Gumbel agent against random and classical MCTS."""

import os
import random as py_random
import sys
import time

from hex_board import Player
from mcts import create_gumbel_mcts
from progress import make_progress

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HexClassic"))


def _make_board(board_size):
    try:
        from chex_board import CHexBoard
        return CHexBoard(board_size)
    except ImportError:
        from hex_board import HexBoard
        return HexBoard(board_size)


class RandomAgent:
    def select_move(self, board, player):
        return py_random.choice(board.get_empty_cells())


def _play_eval_games_batched(config, network, device, num_games, opponent_factory, gumbel_is_black_fn, progress_desc):
    """Run evaluation games in parallel with shared batched NN inference."""
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
            active_games = sum(0 if is_finished else 1 for is_finished in finished)
            progress.set_postfix(moves=moves_played[0], active=active_games)

    def _is_gumbel_turn(game_idx):
        if gumbel_is_black[game_idx]:
            return currents[game_idx] == Player.BLACK
        return currents[game_idx] == Player.WHITE

    def _start_gumbel_turn(game_idx):
        searches[game_idx] = mcts.new_search(int(currents[game_idx]))
        in_search[game_idx] = True

    def _play_move(game_idx, action):
        boards[game_idx].play(action, currents[game_idx])
        moves_played[0] += 1
        _refresh_progress()
        if boards[game_idx].check_win(currents[game_idx]):
            winners[game_idx] = currents[game_idx]
            finished[game_idx] = True
            progress.update(1)
            _refresh_progress(force=True)
            return
        currents[game_idx] = Player.WHITE if currents[game_idx] == Player.BLACK else Player.BLACK

    def _commit_gumbel_move(game_idx):
        action, _ = mcts.finalize_search(
            searches[game_idx],
            select_action_by_count=config.eval_select_action_by_count,
            select_action_by_softmax_count=config.eval_select_action_by_softmax_count,
            temperature=config.eval_select_action_softmax_temperature,
            value_threshold=config.eval_select_action_value_threshold,
        )
        in_search[game_idx] = False
        _play_move(game_idx, action)

    while not all(finished):
        for game_idx in range(num_games):
            if finished[game_idx] or in_search[game_idx]:
                continue
            if _is_gumbel_turn(game_idx):
                _start_gumbel_turn(game_idx)
            else:
                move = opponents[game_idx].select_move(boards[game_idx], currents[game_idx])
                _play_move(game_idx, move)

        eval_requests = []
        for game_idx in range(num_games):
            if finished[game_idx] or not in_search[game_idx]:
                continue

            search = searches[game_idx]
            if search.root.count == 0:
                state, legal_actions = mcts.prepare_expand(
                    search.root,
                    boards[game_idx],
                    int(currents[game_idx]),
                )
                eval_requests.append(("root", game_idx, state, legal_actions, None))
                continue

            if mcts.search_complete(search):
                _commit_gumbel_move(game_idx)
                continue

            leaf_request = mcts.simulate_until_leaf(search, boards[game_idx])
            if leaf_request is None:
                if mcts.search_complete(search):
                    _commit_gumbel_move(game_idx)
                continue

            state, legal_actions = mcts.prepare_expand(
                leaf_request.node,
                leaf_request.board,
                leaf_request.to_play,
            )
            eval_requests.append(("leaf", game_idx, state, legal_actions, leaf_request))

        if not eval_requests:
            continue

        policy_batch, value_batch = mcts.evaluate_states([request[2] for request in eval_requests])
        for idx, (request_type, game_idx, _state, legal_actions, extra) in enumerate(eval_requests):
            policy_logits = policy_batch[idx]
            value = float(value_batch[idx])
            if request_type == "root":
                mcts.finish_root(
                    searches[game_idx],
                    policy_logits,
                    value,
                    legal_actions,
                    add_noise=False,
                )
            else:
                mcts.finish_leaf(extra, policy_logits, value, legal_actions)

    _refresh_progress(force=True)
    progress.close()
    return winners, gumbel_is_black


def evaluate_vs_random(config, network, device, num_games=None):
    if num_games is None:
        num_games = config.eval_games

    winners, gumbel_is_black = _play_eval_games_batched(
        config,
        network,
        device,
        num_games,
        opponent_factory=RandomAgent,
        gumbel_is_black_fn=lambda idx: idx % 2 == 0,
        progress_desc="    eval random",
    )
    wins = 0
    for game_idx in range(num_games):
        if (gumbel_is_black[game_idx] and winners[game_idx] == Player.BLACK) or (
            not gumbel_is_black[game_idx] and winners[game_idx] == Player.WHITE
        ):
            wins += 1
    return 100.0 * wins / num_games


def evaluate_vs_classical_mcts(config, network, device, mcts_sims=None, num_games=None):
    if num_games is None:
        num_games = config.eval_games
    if mcts_sims is None:
        mcts_sims = config.eval_mcts_simulations

    try:
        from cmcts_hex import CMCTSHex
        from mcts_hex import SimulationType
        classic_mcts = CMCTSHex
    except ImportError:
        try:
            from mcts_hex import MCTSHex, SimulationType
            classic_mcts = MCTSHex
        except ImportError:
            print("  Classical MCTS not available (HexClassic not in path), skipping.")
            return None

    def make_opponent():
        return classic_mcts(
            board_size=config.board_size,
            c_uct=0.0,
            rave_bias=0.00025,
            use_rave=True,
            simulation_type=SimulationType.BRIDGES,
            num_simulations=mcts_sims,
        )

    winners, gumbel_is_black = _play_eval_games_batched(
        config,
        network,
        device,
        num_games,
        opponent_factory=make_opponent,
        gumbel_is_black_fn=lambda idx: idx % 2 == 0,
        progress_desc="    eval classic",
    )
    wins = 0
    for game_idx in range(num_games):
        if (gumbel_is_black[game_idx] and winners[game_idx] == Player.BLACK) or (
            not gumbel_is_black[game_idx] and winners[game_idx] == Player.WHITE
        ):
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

    return {"vs_random": wr_random, "vs_mcts": wr_mcts}
