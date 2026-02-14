"""
Interactive Hex game: Human vs MCTS AI.

Usage:
  python play_hex.py [--size N] [--sims N] [--color black|white]
"""

import argparse
import sys
import time

from hex_board import HexBoard, Player
from mcts_hex import MCTSHex, SimulationType


def display_board_fancy(board):
    """Display the board with a nice hex layout and colors."""
    size = board.size

    BLACK_STONE = "\033[1;97;40m B \033[0m"
    WHITE_STONE = "\033[1;30;47m W \033[0m"
    EMPTY_CELL = "\033[90m . \033[0m"

    symbols = {0: EMPTY_CELL, 1: BLACK_STONE, 2: WHITE_STONE}

    print()
    print(f"  \033[1;47;30m WHITE connects LEFT <-> RIGHT \033[0m")
    print(f"  \033[1;40;97m BLACK connects TOP  <-> BOTTOM \033[0m")
    print()

    # Column headers
    header = "    " + "".join(f" {c:X} " for c in range(size))
    print(header)

    for r in range(size):
        indent = " " * (r * 2)
        cells = "".join(symbols[board.board[r * size + c]] for c in range(size))
        label = f"{r:X}"
        print(f"{indent} {label}  {cells}  {label}")

    indent = " " * (size * 2)
    footer = indent + "  " + "".join(f" {c:X} " for c in range(size))
    print(footer)
    print()


def human_move(board, player):
    """Get a move from human player."""
    while True:
        try:
            inp = input(f"  Your move (row col): ").strip()
            if inp.lower() in ('q', 'quit', 'exit'):
                print("Goodbye!")
                sys.exit(0)
            parts = inp.split()
            if len(parts) != 2:
                print("  Enter: row col (e.g., '5 3')")
                continue
            r = int(parts[0], 16) if any(c.isalpha() for c in parts[0]) else int(parts[0])
            c = int(parts[1], 16) if any(c.isalpha() for c in parts[1]) else int(parts[1])
            if not (0 <= r < board.size and 0 <= c < board.size):
                print(f"  Out of bounds. Use 0-{board.size-1}.")
                continue
            idx = r * board.size + c
            if board.board[idx] != 0:
                print("  Cell is occupied!")
                continue
            return idx
        except (ValueError, KeyboardInterrupt):
            print("\n  Invalid input. Enter: row col")


def play_interactive(size=7, num_sims=3000, human_color='black'):
    """Play an interactive game against the AI."""
    board = HexBoard(size)

    ai = MCTSHex(
        board_size=size,
        c_uct=0.0,
        rave_bias=0.00025,
        use_rave=True,
        simulation_type=SimulationType.BRIDGES,
        num_simulations=num_sims,
    )

    human_player = Player.BLACK if human_color == 'black' else Player.WHITE
    ai_player = human_player.opponent

    print(f"\n{'='*50}")
    print(f"  HEX GAME: {size}x{size}")
    print(f"  You: {'BLACK (TOP-BOTTOM)' if human_player == Player.BLACK else 'WHITE (LEFT-RIGHT)'}")
    print(f"  AI: {num_sims} MCTS simulations")
    print(f"  Enter 'row col' (e.g. '5 3'), 'q' to quit")
    print(f"{'='*50}")

    current = Player.BLACK
    move_num = 0

    while True:
        display_board_fancy(board)

        if current == human_player:
            print(f"  Move {move_num + 1}: Your turn")
            move = human_move(board, current)
        else:
            print(f"  Move {move_num + 1}: AI thinking...")
            start = time.time()
            move = ai.select_move(board, current)
            elapsed = time.time() - start
            r, c = board.idx_to_rc(move)
            print(f"  AI plays: {r} {c} ({elapsed:.1f}s)")

        board.play(move, int(current))
        move_num += 1

        if board.check_win(int(current)):
            display_board_fancy(board)
            if current == human_player:
                print("  YOU WIN!")
            else:
                print("  AI wins. Better luck next time!")
            break

        current = current.opponent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play Hex against MCTS AI')
    parser.add_argument('--size', type=int, default=7, help='Board size (default: 7)')
    parser.add_argument('--sims', type=int, default=3000, help='MCTS simulations (default: 3000)')
    parser.add_argument('--color', choices=['black', 'white'], default='black', help='Your color')
    args = parser.parse_args()
    play_interactive(args.size, args.sims, args.color)
