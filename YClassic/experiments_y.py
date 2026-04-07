"""
Comparative experiments: MCTS behavior on the Game of Y.

Runs the same experimental protocol as "Monte-Carlo Hex"
(Cazenave & Saffidine), adapted to Y. Since Y does not admit
well-defined bridge templates, all experiments use random
playouts to isolate the effect of the core UCT + RAVE algorithm.

Experiments:
  Table 1: Effect of simulation count vs reference agent
  Table 2: UCT constant variation with RAVE
  Table 3: RAVE vs pure UCT (no RAVE)
  Table 4: RAVE bias variation
"""

import time
import random
import multiprocessing
import sys

from y_board import YBoard, Player
from mcts_y import MCTSY


NUM_WORKERS = multiprocessing.cpu_count()


# ---------------------------
# Game logic
# ---------------------------

def play_game(size, black_agent, white_agent):
    board = YBoard(size)
    current = Player.BLACK

    while True:
        if current == Player.BLACK:
            move = black_agent.select_move(board, current)
        else:
            move = white_agent.select_move(board, current)

        board.play(move, int(current))

        if board.check_win(int(current)):
            return current

        current = current.opponent


# ---------------------------
# Worker
# ---------------------------

def _play_single_game(args):
    game_idx, size, p1, p2, seed = args
    random.seed(seed)

    agent1 = MCTSY(board_size=size, **p1)
    agent2 = MCTSY(board_size=size, **p2)

    if game_idx % 2 == 0:
        winner = play_game(size, agent1, agent2)
        return 1 if winner == Player.BLACK else 0
    else:
        winner = play_game(size, agent2, agent1)
        return 1 if winner == Player.WHITE else 0


# ---------------------------
# Core experiment runner
# ---------------------------

def run_experiment(size, p1, p2, num_games=50, desc=""):
    print(f"\n{desc}")
    print(f"Agent1: {p1}")
    print(f"Agent2: {p2}")

    start = time.time()

    seeds = [random.randrange(2**31) for _ in range(num_games)]
    tasks = [(i, size, p1, p2, seeds[i]) for i in range(num_games)]

    wins = 0

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_play_single_game, tasks)):
            wins += result
            if (i + 1) % max(1, num_games // 5) == 0:
                pct = 100 * wins / (i + 1)
                print(f"  [{i+1}/{num_games}] {pct:.1f}%")

    pct = 100 * wins / num_games
    elapsed = time.time() - start

    print(f"RESULT: {wins}/{num_games} = {pct:.1f}% ({elapsed:.1f}s)")
    return pct


# ---------------------------
# Experiments
# ---------------------------

def quick_sanity():
    print("\nSanity: MCTS vs Random")

    class RandomAgent:
        def select_move(self, board, player):
            return random.choice(board.get_empty_cells())

    size = 5
    games = 20

    wins = 0
    for i in range(games):
        mcts = MCTSY(num_simulations=500)
        rand = RandomAgent()

        if i % 2 == 0:
            winner = play_game(size, mcts, rand)
            wins += (winner == Player.BLACK)
        else:
            winner = play_game(size, rand, mcts)
            wins += (winner == Player.WHITE)

    print(f"MCTS wins: {wins}/{games} = {100*wins/games:.1f}%")


def table1():
    print("\nTable 1: simulation count")

    ref = {'num_simulations': 5000, 'c_uct': 0.5}

    for sims in [500, 1000, 2000, 10000]:
        test = {**ref, 'num_simulations': sims}
        run_experiment(5, test, ref, 40, f"{sims} vs 5000 sims")


def table2():
    print("\nTable 2: UCT constant")

    base = {'num_simulations': 5000}
    ref = {**base, 'c_uct': 0.5}

    for c in [0.0, 0.2, 0.5, 1.0]:
        test = {**base, 'c_uct': c}
        run_experiment(5, test, ref, 40, f"C={c} vs 0.5")


def table3():
    print("\nTable 3: RAVE vs no-RAVE")

    rave = {
        'num_simulations': 5000,
        'use_rave': True,
        'c_uct': 0.0
    }

    no_rave = {
        'num_simulations': 5000,
        'use_rave': False,
        'c_uct': 0.5
    }

    run_experiment(5, rave, no_rave, 40, "RAVE vs no-RAVE")


def table4():
    print("\nTable 4: RAVE bias")

    base = {
        'num_simulations': 5000,
        'use_rave': True,
        'c_uct': 0.0
    }

    ref = {**base, 'rave_bias': 0.001}

    for b in [0.0005, 0.00025, 0.000125]:
        test = {**base, 'rave_bias': b}
        run_experiment(5, test, ref, 40, f"bias={b}")


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sanity"

    if mode == "sanity":
        quick_sanity(seed)
    elif mode == "table1":
        table1()
    elif mode == "table2":
        table2()
    elif mode == "table3":
        table3()
    elif mode == "table4":
        table4()
    elif mode == "all":
        table1()
        table2()
        table3()
        table4()
    else:
        print("Usage: python experiments_y.py [sanity|table1|table2|table3|table4|all]") 