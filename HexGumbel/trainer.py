"""Training loop: self-play -> train -> evaluate."""

import os
import time
import torch
import torch.nn.functional as F

from config import GumbelZeroConfig
from neural_net import HexNet
from self_play import ReplayBuffer, generate_self_play_data
from evaluate import run_evaluation


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        self.network = HexNet(
            board_size=config.board_size,
            num_channels=config.num_channels,
            num_res_blocks=config.num_res_blocks,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = config.num_iterations * config.training_steps_per_iteration
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )

        self.buffer = ReplayBuffer(config.replay_buffer_capacity, config.board_size)
        self.checkpoint_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoints"
        )

    def train(self, start_iteration=0):
        for iteration in range(start_iteration, self.config.num_iterations):
            print()
            print(f"  Iteration {iteration + 1}/{self.config.num_iterations}")

            # Self-play
            t0 = time.time()
            games = generate_self_play_data(
                self.config, self.network,
                self.config.num_self_play_games_per_iteration,
                self.device,
            )
            for game_data in games:
                self.buffer.add_game(game_data)
            positions = sum(len(g) for g in games)
            print(f"  Self-play: {len(games)} games, {positions} positions "
                  f"({time.time() - t0:.1f}s), buffer: {len(self.buffer)}")

            # Training
            if len(self.buffer) < self.config.batch_size:
                print("  Buffer too small, skipping training.")
                continue

            t0 = time.time()
            self.network.train()
            metrics = self._train_epoch()
            print(f"  Training: {self.config.training_steps_per_iteration} steps "
                  f"({time.time() - t0:.1f}s), "
                  f"policy_loss={metrics['policy_loss']:.4f}, "
                  f"value_loss={metrics['value_loss']:.4f}")

            # Evaluation
            is_last = (iteration + 1 == self.config.num_iterations)
            if (iteration + 1) % self.config.eval_interval == 0 or is_last:
                run_evaluation(self.config, self.network, self.device, iteration + 1)

            # Checkpoint
            if (iteration + 1) % self.config.checkpoint_interval == 0 or is_last:
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
            policy_loss = F.kl_div(log_pred, target_policies, reduction='batchmean', log_target=False)
            value_loss = F.mse_loss(pred_values, target_values)
            loss = policy_loss + self.config.value_loss_weight * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_policy += policy_loss.item()
            total_value += value_loss.item()

        return {'policy_loss': total_policy / steps, 'value_loss': total_value / steps}

    def _save_checkpoint(self, iteration):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"iter_{iteration:04d}.pt")
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'buffer_size': len(self.buffer),
        }, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print(f"  Loaded checkpoint: {path} (iteration {ckpt['iteration']})")
        return ckpt['iteration']
