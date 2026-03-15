"""Benchmark Gumbel AlphaZero against random and classical MCTS."""

import sys
import os
import time

from hex_board import Player

# Add HexClassic to path for classical MCTS imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'HexClassic'))

from mcts import GumbelZeroAgent


def _make_board(board_size):
    try:
        from chex_board import CHexBoard
        return CHexBoard(board_size)
    except ImportError:
        from hex_board import HexBoard
        return HexBoard(board_size)


def _play_game(board_size, black_agent, white_agent, force_python_board=False):
    """Minimal game loop. Uses Cython board when possible, falls back to
    pure-Python when needed (e.g. classical MCTS needs .board[i])."""
    if force_python_board:
        from hex_board import HexBoard
        board = HexBoard(board_size)
    else:
        board = _make_board(board_size)

    current = 1  # BLACK
    agents = {1: black_agent, 2: white_agent}

    while True:
        move = agents[current].select_move(board, current)
        board.play(move, current)
        if board.check_win(current):
            return Player(current)
        current = 3 - current


class RandomAgent:
    """Picks a random empty cell."""
    def select_move(self, board, player):
        import random
        return random.choice(board.get_empty_cells())


def evaluate_vs_random(config, network, device, num_games=None):
    if num_games is None:
        num_games = config.eval_games

    agent = GumbelZeroAgent(config, network, device)
    rand = RandomAgent()
    wins = 0

    for i in range(num_games):
        if i % 2 == 0:
            winner = _play_game(config.board_size, black_agent=agent, white_agent=rand)
            if winner == Player.BLACK:
                wins += 1
        else:
            winner = _play_game(config.board_size, black_agent=rand, white_agent=agent)
            if winner == Player.WHITE:
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
        ClassicMCTS = CMCTSHex
    except ImportError:
        try:
            from mcts_hex import MCTSHex, SimulationType
            ClassicMCTS = MCTSHex
        except ImportError:
            print("  Classical MCTS not available (HexClassic not in path), skipping.")
            return None

    agent = GumbelZeroAgent(config, network, device)
    mcts_agent = ClassicMCTS(
        board_size=config.board_size,
        c_uct=0.0,
        rave_bias=0.00025,
        use_rave=True,
        simulation_type=SimulationType.BRIDGES,
        num_simulations=mcts_sims,
    )
    wins = 0

    for i in range(num_games):
        if i % 2 == 0:
            winner = _play_game(config.board_size, black_agent=agent, white_agent=mcts_agent)
            if winner == Player.BLACK:
                wins += 1
        else:
            winner = _play_game(config.board_size, black_agent=mcts_agent, white_agent=agent)
            if winner == Player.WHITE:
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

    return {'vs_random': wr_random, 'vs_mcts': wr_mcts}
