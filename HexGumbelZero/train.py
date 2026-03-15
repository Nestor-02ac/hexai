"""
Train Gumbel AlphaZero for Hex.

Usage:
  python train.py                                # defaults (7x7, 100 iters)
  python train.py --board-size 9 --channels 128  # larger board
  python train.py --resume checkpoints/iter_0050.pt
  python train.py --iterations 5 --games-per-iter 10  # quick smoke test
"""

import argparse
import random
import numpy as np
import torch

from config import GumbelZeroConfig
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train Gumbel AlphaZero for Hex')
    parser.add_argument('--board-size', type=int, default=7)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--res-blocks', type=int, default=5)
    parser.add_argument('--simulations', type=int, default=0,
                        help='MCTS sims per move (0 = auto: board_size^2)')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--games-per-iter', type=int, default=100)
    parser.add_argument('--eval-games', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = GumbelZeroConfig(
        board_size=args.board_size,
        num_channels=args.channels,
        num_res_blocks=args.res_blocks,
        num_simulations=args.simulations,
        learning_rate=args.lr,
        num_iterations=args.iterations,
        num_self_play_games_per_iteration=args.games_per_iter,
        eval_games=args.eval_games,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )

    print(f"  Gumbel AlphaZero for {config.board_size}x{config.board_size} Hex")
    print(f"  Network: {config.num_channels}ch, {config.num_res_blocks} res blocks")
    print(f"  MCTS: {config.num_simulations} sims, {config.max_considered_actions} candidates")
    print(f"  Training: {config.num_iterations} iters, {config.num_self_play_games_per_iteration} games/iter")
    print(f"  Device: {config.device}")

    trainer = Trainer(config)

    start_iter = 0
    if args.resume:
        start_iter = trainer.load_checkpoint(args.resume)

    trainer.train(start_iteration=start_iter)


if __name__ == '__main__':
    main()
