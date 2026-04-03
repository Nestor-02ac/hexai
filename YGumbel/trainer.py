"""
Training loop: self-play -> train -> evaluate (Y version).
"""

import json
import os
import platform
import socket
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from config import GumbelZeroConfig
from neural_net import YNet 
from progress import make_progress
from self_play import ReplayBuffer, generate_self_play_data
from evaluate import run_evaluation


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.project_dir = Path(__file__).resolve().parent
        self.checkpoint_root = self.project_dir / "checkpoints"

        self.checkpoint_dir = None
        self.run_id = None
        self.run_metadata_path = None
        self.history_path = None
        self.run_metadata = None
        self.self_play_position_estimate = None

        # Y network
        self.network = YNet(
            board_size=config.board_size,
            num_channels=config.num_channels,
            num_res_blocks=config.num_res_blocks,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        if getattr(config, 'use_lr_scheduler', True):
            total_steps = config.num_iterations * config.training_steps_per_iteration
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )
        else:
            self.scheduler = None

        self.buffer = ReplayBuffer(config.replay_buffer_capacity, config.board_size)

    def configure_run(self, cli_args=None, argv=None, resume_path=None):
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        resume_path_obj = Path(resume_path).resolve() if resume_path else None
        self.checkpoint_dir = self._resolve_run_dir(resume_path_obj)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = self.checkpoint_dir.name
        self.run_metadata_path = self.checkpoint_dir / "run.json"
        self.history_path = self.checkpoint_dir / "history.jsonl"

        existing_metadata = None
        if self.run_metadata_path.exists():
            with self.run_metadata_path.open("r", encoding="utf-8") as fh:
                existing_metadata = json.load(fh)

        self.run_metadata = self._build_run_metadata(
            cli_args=cli_args or {},
            argv=argv or [],
            resume_path=resume_path_obj,
            existing_metadata=existing_metadata,
        )
        self._write_json(self.run_metadata_path, self.run_metadata)
        self._write_latest_pointers(resume_path_obj)

        print(f"  Checkpoints: {self.checkpoint_dir}")

    def train(self, start_iteration=0):
        for iteration in range(start_iteration, self.config.num_iterations):
            print()
            print(f"  Iteration {iteration + 1}/{self.config.num_iterations}")

            # Self-play
            t0 = time.time()
            games = generate_self_play_data(
                self.config,
                self.network,
                self.config.num_self_play_games_per_iteration,
                self.device,
                progress_total=self._estimate_self_play_positions(),
            )

            for game_data in games:
                self.buffer.add_game(game_data)

            positions = sum(len(g) for g in games)
            self._update_self_play_position_estimate(positions)

            print(f"  Self-play: {len(games)} games, {positions} positions, buffer: {len(self.buffer)}")

            # Training
            if len(self.buffer) < self.config.batch_size:
                print("  Buffer too small, skipping training.")
                continue

            self.network.train()
            metrics = self._train_epoch()

            print(
                f"  Training: policy_loss={metrics['policy_loss']:.4f}, "
                f"value_loss={metrics['value_loss']:.4f}"
            )

            # Evaluation
            if (iteration + 1) % self.config.eval_interval == 0:
                run_evaluation(self.config, self.network, self.device, iteration + 1)

            # Checkpoint
            if (iteration + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(iteration + 1)

    def _train_epoch(self):
        total_policy = 0.0
        total_value = 0.0
        steps = self.config.training_steps_per_iteration

        for _ in range(steps):
            states, target_policies, target_values = self.buffer.sample(self.config.batch_size)

            states = states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)

            policy_logits, pred_values = self.network(states)
            pred_values = pred_values.squeeze(-1)

            log_pred = F.log_softmax(policy_logits, dim=-1)
            policy_loss = F.kl_div(log_pred, target_policies, reduction='batchmean')
            value_loss = F.mse_loss(pred_values, target_values)

            loss = policy_loss + self.config.value_loss_weight * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_policy += policy_loss.item()
            total_value += value_loss.item()

        return {
            'policy_loss': total_policy / steps,
            'value_loss': total_value / steps,
        }


    def _estimate_self_play_positions(self):
        num_cells = self.config.board_size * (self.config.board_size + 1) // 2 

        return max(
            self.config.num_self_play_games_per_iteration,
            int(round(0.80 * self.config.num_self_play_games_per_iteration * num_cells)),
        )

    def _update_self_play_position_estimate(self, positions):
        if self.self_play_position_estimate is None:
            self.self_play_position_estimate = float(positions)
            return
        self.self_play_position_estimate = 0.7 * self.self_play_position_estimate + 0.3 * float(positions)

    def _save_checkpoint(self, iteration):
        path = self.checkpoint_dir / f"iter_{iteration:04d}.pt"

        torch.save({
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

        print(f"  Checkpoint saved: {path}")

    def _resolve_run_dir(self, resume_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.checkpoint_root / f"{timestamp}_Y"

    def _build_run_metadata(self, cli_args, argv, resume_path, existing_metadata=None):
        return {
            "project": "YGumbel",
            "config": asdict(self.config),
        }

    @staticmethod
    def _write_json(path, payload):
        with Path(path).open("w") as f:
            json.dump(payload, f, indent=2)

    def _write_latest_pointers(self, path):
        pass