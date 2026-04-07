"""
Generate expert dataset for Y:
Runs MCTS games and saves (board state, expert move, player) tuples.

Output: expert_data_y.jsonl (one JSON per move)

Usage:
  python generate_expert_data_y.py
  python generate_expert_data_y.py --games 1000 --sims 10000
  python generate_expert_data_y.py --games 200 --sims 16000 --output ../YGumbel/expert_data_16k.jsonl
"""

import argparse
import json
import time
from y_board import YBoard, Player
from mcts_y import MCTSY


def board_to_list(board: YBoard):
    return list(board.board)


def main():
    parser = argparse.ArgumentParser(description='Generate expert MCTS data for Y')
    parser.add_argument('--games', type=int, default=1000, help='Number of games')
    parser.add_argument('--sims', type=int, default=1000, help='MCTS simulations per move')
    parser.add_argument('--board-size', type=int, default=9, help='Triangle size')
    parser.add_argument('--output', type=str, default='expert_data_y.jsonl', help='Output file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random_seed = args.seed

    print(f"  Generating {args.games} Y games with {args.sims} sims")
    print(f"  Board size: {args.board_size}")
    print(f"  Output: {args.output}")

    total_positions = 0
    t0 = time.time()

    with open(args.output, "w") as f:
        for game_idx in range(args.games):
            game_t0 = time.time()

            board = YBoard(args.board_size)

            mcts = MCTSY(
                board_size=args.board_size,
                num_simulations=args.sims,
                use_rave=True,
                c_uct=0.0,
                rave_bias=0.00025,
            )

            states = []
            moves = []
            players = []

            current_player = Player.BLACK

            while True:
                # Save state BEFORE move (same as Hex)
                state = board_to_list(board)

                move = mcts.select_move(board, current_player)

                states.append(state)
                moves.append(move)
                players.append(int(current_player))

                success = board.play(move, current_player)
                assert success, f"Illegal move {move}"

                if board.check_win(current_player):
                    break

                current_player = current_player.opponent

            # Save all moves from this game
            for s, m, p in zip(states, moves, players):
                f.write(json.dumps({
                    "board": s,
                    "move": m,
                    "player": p
                }) + "\n")

            total_positions += len(states)

            elapsed = time.time() - game_t0
            total_elapsed = time.time() - t0
            avg_per_game = total_elapsed / (game_idx + 1)
            eta = avg_per_game * (args.games - game_idx - 1)

            print(f"  Game {game_idx+1}/{args.games} — {len(states)} moves, "
                  f"{elapsed:.1f}s, total={total_positions} pos, ETA={eta:.0f}s")

    total_time = time.time() - t0
    print(f"  Done: {total_positions} positions in {total_time:.1f}s — saved to {args.output}")


if __name__ == "__main__":
    main()