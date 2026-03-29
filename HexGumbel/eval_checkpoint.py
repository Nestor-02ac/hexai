"""
Load a checkpoint (model state dict or full checkpoint) and run evaluation
against Random and Classical MCTS to produce a baseline before self-play.

Usage example (fast baseline):
  conda run -n hex python HexGumbel/eval_checkpoint.py \
    --checkpoint checkpoints/pretrained_model_16k.pt --board-size 7 --channels 192 --res-blocks 20 \
    --games 40 --mcts-sims 1000

"""
import argparse
import random
import numpy as np
import torch
from config import GumbelZeroConfig
from neural_net import HexNet
from evaluate import run_evaluation
from pathlib import Path


def load_model_from_checkpoint(path, config, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")
    ckpt = torch.load(str(path), map_location=device)
    model = HexNet(config.board_size, num_channels=config.num_channels, num_res_blocks=config.num_res_blocks)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        # assume ckpt is a plain state_dict
        model.load_state_dict(ckpt)
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
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
    parser.add_argument('--games', type=int, default=40)
    parser.add_argument('--mcts-sims', type=int, default=1000)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
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
    )
    config.eval_games = args.games
    config.eval_mcts_simulations = args.mcts_sims
    if args.device:
        config.device = args.device

    device = torch.device(config.device)
    network = load_model_from_checkpoint(args.checkpoint, config, device)

    # Run evaluation (iteration labelled 0 for baseline)
    run_evaluation(config, network, device, iteration=0)


if __name__ == '__main__':
    main()
