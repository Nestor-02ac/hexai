"""
Generate expert dataset for supervised pretraining:
Runs MCTS games and saves (board state, expert move, player) tuples for each move.
Output: expert_data.jsonl (one JSON per move)

Usage:
  python generate_expert_data.py                              # defaults (1000 games, 1000 sims)
  python generate_expert_data.py --games 1000 --sims 16000    # 16K-sim expert data
  python generate_expert_data.py --games 200 --sims 16000 --output ../HexGumbel/data/expert_data_16k.jsonl
"""

import argparse
import json
import time
from hex_board import HexBoard, Player
from cmcts_hex import CMCTSHex, seed_rng


def board_to_list(board: HexBoard):
    return list(board.board)


def main():
    parser = argparse.ArgumentParser(description='Generate expert MCTS data for pretraining')
    parser.add_argument('--games', type=int, default=1000, help='Number of games to generate')
    parser.add_argument('--sims', type=int, default=1000, help='MCTS simulations per move')
    parser.add_argument('--board-size', type=int, default=7, help='Board size')
    parser.add_argument('--output', type=str, default='expert_data.jsonl', help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed for Cython MCTS')
    args = parser.parse_args()

    seed_rng(args.seed)

    print(f"  Generating {args.games} games with {args.sims} sims on {args.board_size}x{args.board_size} board")
    print(f"  Backend: Cython CMCTSHex")
    print(f"  Output: {args.output}")

    total_positions = 0
    t0 = time.time()

    with open(args.output, "w") as f:
        for game_idx in range(args.games):
            game_t0 = time.time()
            board = HexBoard(args.board_size)
            mcts = CMCTSHex(
                board_size=args.board_size,
                num_simulations=args.sims,
                use_rave=True,
                c_uct=0.0,
                rave_bias=0.00025,
                simulation_type=2,  # BRIDGES
            )
            states = []
            moves = []
            players = []
            current_player = Player.BLACK
            while not board.check_win(Player.BLACK) and not board.check_win(Player.WHITE):
                state = board_to_list(board)
                move = mcts.select_move(board, current_player)
                states.append(state)
                moves.append(move)
                players.append(int(current_player))
                board.play(move, int(current_player))
                current_player = current_player.opponent
            # Save all moves from this game
            for s, m, p in zip(states, moves, players):
                f.write(json.dumps({"board": s, "move": m, "player": p}) + "\n")
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
