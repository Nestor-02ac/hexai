"""
Interactive Y game: Human vs MCTS AI.

Usage:
  python play_y.py [--size N] [--sims N] [--color black|white] [--cython]
"""

import argparse
import sys
import time

try:
    from y_board import Player, YBoard
    from mcts_y import MCTSY, SimulationType
except ModuleNotFoundError:
    from .y_board import Player, YBoard
    from .mcts_y import MCTSY, SimulationType


def _build_agent(size, num_sims, use_cython):
    if use_cython:
        try:
            from cmcts_y import CMCTSY
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Cython backend not built. Run `python3 setup.py build_ext --inplace` in YClassic/."
            ) from exc

        return CMCTSY(
            board_size=size,
            c_uct=0.0,
            rave_bias=0.00025,
            use_rave=True,
            simulation_type=SimulationType.CONNECTIVITY,
            num_simulations=num_sims,
        )

    return MCTSY(
        board_size=size,
        c_uct=0.0,
        rave_bias=0.00025,
        use_rave=True,
        simulation_type=SimulationType.CONNECTIVITY,
        num_simulations=num_sims,
    )


def display_board_fancy(board):
    """Display the Y board with side labels and ANSI colors."""
    size = board.size

    black_stone = "\033[1;97;40m B \033[0m"
    white_stone = "\033[1;30;47m W \033[0m"
    empty_cell = "\033[90m . \033[0m"

    symbols = {0: empty_cell, 1: black_stone, 2: white_stone}

    print()
    print("  Y GAME")
    print("  Goal: connect LEFT, RIGHT, and BOTTOM in one group")
    print()

    for r in range(size):
        indent = " " * (size - r - 1)
        row = []
        for c in range(r + 1):
            idx = board.rc_to_idx(r, c)
            row.append(symbols[board.get_cell(idx) if hasattr(board, "get_cell") else board.board[idx]])
        print(indent + " ".join(row))
    print()


def human_move(board):
    """Read and validate a human move."""
    while True:
        try:
            inp = input("  Your move (row col): ").strip()
            if inp.lower() in ("q", "quit", "exit"):
                print("Goodbye!")
                sys.exit(0)

            parts = inp.split()
            if len(parts) != 2:
                print("  Enter: row col (e.g. '3 1').")
                continue

            r = int(parts[0], 16) if any(ch.isalpha() for ch in parts[0]) else int(parts[0])
            c = int(parts[1], 16) if any(ch.isalpha() for ch in parts[1]) else int(parts[1])

            if not (0 <= r < board.size and 0 <= c <= r):
                print(f"  Invalid position: require 0 <= col <= row < {board.size}.")
                continue

            idx = board.rc_to_idx(r, c)
            cell = board.get_cell(idx) if hasattr(board, "get_cell") else board.board[idx]
            if cell != 0:
                print("  Cell is occupied.")
                continue

            return idx

        except (ValueError, KeyboardInterrupt):
            print("\n  Invalid input. Enter: row col")


def play_interactive(size=7, num_sims=3000, human_color="black", use_cython=False):
    """Play an interactive game against the AI."""
    board = YBoard(size)
    ai = _build_agent(size, num_sims, use_cython)

    human_player = Player.BLACK if human_color == "black" else Player.WHITE

    print()
    print(f"  Y GAME: side length {size}")
    print(f"  You: {'BLACK' if human_player == Player.BLACK else 'WHITE'}")
    print(f"  AI: {num_sims} MCTS simulations ({'Cython' if use_cython else 'Python'})")
    print("  Enter 'row col' or 'q' to quit")

    current = Player.BLACK
    move_num = 0

    while True:
        display_board_fancy(board)

        if current == human_player:
            print(f"  Move {move_num + 1}: your turn")
            move = human_move(board)
        else:
            print(f"  Move {move_num + 1}: AI thinking...")
            start = time.time()
            move = ai.select_move(board, current)
            elapsed = time.time() - start
            r, c = board.idx_to_rc(move)
            print(f"  AI plays: {r} {c} ({elapsed:.2f}s)")

        board.play(move, int(current))
        move_num += 1

        if board.check_win(int(current)):
            display_board_fancy(board)
            if current == human_player:
                print("  YOU WIN!")
            else:
                print("  AI wins!")
            break

        current = current.opponent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Y against MCTS AI")
    parser.add_argument("--size", type=int, default=7, help="Board side length (default: 7)")
    parser.add_argument("--sims", type=int, default=3000, help="MCTS simulations (default: 3000)")
    parser.add_argument("--color", choices=["black", "white"], default="black", help="Your color")
    parser.add_argument("--cython", action="store_true", help="Use the Cython backend")
    args = parser.parse_args()

    play_interactive(args.size, args.sims, args.color, args.cython)
