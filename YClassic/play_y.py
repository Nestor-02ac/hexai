import argparse
import sys
import time

from y_board import YBoard, Player
from mcts_y import MCTSY


def display_board_fancy(board):
    """Display triangular Y board."""
    size = board.size

    BLACK_STONE = "\033[1;97;40m B \033[0m"
    WHITE_STONE = "\033[1;30;47m W \033[0m"
    EMPTY_CELL = "\033[90m . \033[0m"

    symbols = {0: EMPTY_CELL, 1: BLACK_STONE, 2: WHITE_STONE}

    print()
    print("      Y GAME")
    print("  Connect ALL THREE SIDES")
    print()

    for r in range(size):
        indent = " " * (size - r - 1)
        row = []
        for c in range(r + 1):
            idx = board.rc_to_idx(r, c)
            row.append(symbols[board.board[idx]])
        print(indent + " ".join(row))
    print()


def human_move(board, player):
    """Get a move from human player."""
    while True:
        try:
            inp = input("  Your move (row col): ").strip()
            if inp.lower() in ('q', 'quit', 'exit'):
                print("Goodbye!")
                sys.exit(0)

            parts = inp.split()
            if len(parts) != 2:
                print("  Enter: row col (e.g., '3 1')")
                continue

            r = int(parts[0])
            c = int(parts[1])

            if not (0 <= r < board.size and 0 <= c <= r):
                print(f"  Invalid position: must satisfy 0 ≤ col ≤ row < {board.size}")
                continue

            idx = board.rc_to_idx(r, c)

            if board.board[idx] != 0:
                print("  Cell is occupied!")
                continue

            return idx

        except (ValueError, KeyboardInterrupt):
            print("\n  Invalid input. Enter: row col")


def play_interactive(size=5, num_sims=5000, human_color='black'):
    """Play Y against MCTS AI."""
    board = YBoard(size)

    ai = MCTSY(
        board_size=size,
        c_uct=0.5,
        num_simulations=num_sims
    )

    human_player = Player.BLACK if human_color == 'black' else Player.WHITE
    ai_player = human_player.opponent

    print()
    print(f"  Y GAME (size={size})")
    print(f"  You: {'BLACK' if human_player == Player.BLACK else 'WHITE'}")
    print(f"  AI simulations: {num_sims}")
    print("  Enter 'row col', or 'q' to quit")

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play Y against MCTS AI')
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--sims', type=int, default=5000)
    parser.add_argument('--color', choices=['black', 'white'], default='black')
    args = parser.parse_args()

    play_interactive(args.size, args.sims, args.color)