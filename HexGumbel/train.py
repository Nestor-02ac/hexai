"""
Train the Hex Gumbel AlphaZero variant.

Usage:
  python train.py                                # defaults (7x7, 100 iters)
  python train.py --board-size 9 --channels 128  # larger board
    python train.py --simulations 16 --batch-size 1024 --train-steps 220
  python train.py --resume checkpoints/<run_id>/iter_0050.pt
  python train.py --iterations 5 --games-per-iter 10  # quick smoke test
"""

import argparse
import random
import sys
import numpy as np
import torch

from config import GumbelZeroConfig
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train the Hex Gumbel AlphaZero variant')
    parser.add_argument('--board-size', type=int, default=7)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--res-blocks', type=int, default=5)
    parser.add_argument('--simulations', type=int, default=0,
                        help='MCTS sims per move (0 = auto: board_size^2)')
    parser.add_argument('--gumbel-sample-size', type=int, default=0,
                        help='root sample size (0 = auto: num_simulations)')
    parser.add_argument('--gumbel-sigma-visit-c', type=float, default=50.0)
    parser.add_argument('--gumbel-sigma-scale-c', type=float, default=1.0)
    parser.add_argument('--pb-c-base', type=float, default=19652.0)
    parser.add_argument('--pb-c-init', type=float, default=1.25)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--train-steps', type=int, default=200,
                        help='Training updates per outer iteration')
    parser.add_argument('--value-loss-weight', type=float, default=1.0)
    parser.add_argument('--replay-capacity', type=int, default=100_000)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--games-per-iter', type=int, default=100)
    parser.add_argument('--checkpoint-interval', type=int, default=5)
    parser.add_argument('--eval-games', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=5)
    parser.add_argument('--eval-mcts-simulations', type=int, default=1000)
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
        gumbel_sample_size=args.gumbel_sample_size,
        gumbel_sigma_visit_c=args.gumbel_sigma_visit_c,
        gumbel_sigma_scale_c=args.gumbel_sigma_scale_c,
        pb_c_base=args.pb_c_base,
        pb_c_init=args.pb_c_init,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        training_steps_per_iteration=args.train_steps,
        value_loss_weight=args.value_loss_weight,
        replay_buffer_capacity=args.replay_capacity,
        num_iterations=args.iterations,
        num_self_play_games_per_iteration=args.games_per_iter,
        checkpoint_interval=args.checkpoint_interval,
        eval_games=args.eval_games,
        eval_interval=args.eval_interval,
        eval_mcts_simulations=args.eval_mcts_simulations,
        seed=args.seed,
    )

    print(f"  Hex Gumbel AlphaZero for {config.board_size}x{config.board_size} Hex")
    print(f"  Network: {config.num_channels}ch, {config.num_res_blocks} res blocks")
    print(f"  MCTS: {config.num_simulations} sims, {config.gumbel_sample_size} candidates")
    print(
        f"  PUCT/sigma: pb_c=({config.pb_c_base}, {config.pb_c_init}), "
        f"sigma_visit_c={config.gumbel_sigma_visit_c}, sigma_scale_c={config.gumbel_sigma_scale_c}"
    )
    print(
        f"  Training: {config.num_iterations} iters, "
        f"{config.num_self_play_games_per_iteration} games/iter, "
        f"batch={config.batch_size}, steps/iter={config.training_steps_per_iteration}"
    )
    print(
        f"  Optimizer: lr={config.learning_rate}, wd={config.weight_decay}, "
        f"value_loss_weight={config.value_loss_weight}, replay={config.replay_buffer_capacity}"
    )
    print(
        f"  Eval: every {config.eval_interval} iters, {config.eval_games} games, "
        f"opponent_mcts_sims={config.eval_mcts_simulations}"
    )
    print(f"  Device: {config.device}")

    trainer = Trainer(config)
    trainer.configure_run(
        cli_args=vars(args),
        argv=sys.argv,
        resume_path=args.resume,
    )

    start_iter = 0
    if args.resume:
        start_iter = trainer.load_checkpoint(args.resume)

    trainer.train(start_iteration=start_iter)


if __name__ == '__main__':
    main()
