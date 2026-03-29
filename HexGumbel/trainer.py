"""Training loop: self-play -> train -> evaluate."""

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
from neural_net import HexNet
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
        # Optionally use a cosine LR scheduler
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
            self_play_profile = None
            games_result = generate_self_play_data(
                self.config,
                self.network,
                self.config.num_self_play_games_per_iteration,
                self.device,
                progress_total=self._estimate_self_play_positions(),
                return_profile=self.config.profile_self_play,
            )
            if self.config.profile_self_play:
                games, self_play_profile = games_result
            else:
                games = games_result
            for game_data in games:
                self.buffer.add_game(game_data)
            positions = sum(len(g) for g in games)
            self._update_self_play_position_estimate(positions)
            self_play_seconds = time.time() - t0
            print(f"  Self-play: {len(games)} games, {positions} positions "
                  f"({self_play_seconds:.1f}s), buffer: {len(self.buffer)}")
            if self_play_profile is not None:
                print(f"  Self-play profile: {self._format_self_play_profile(self_play_profile, self_play_seconds)}")

            iteration_record = {
                "iteration": iteration + 1,
                "timestamp": self._timestamp(),
                "buffer_size": len(self.buffer),
                "self_play": {
                    "games": len(games),
                    "positions": positions,
                    "seconds": round(self_play_seconds, 4),
                    "profile": self._profile_record(self_play_profile),
                },
                "training": None,
                "evaluation": None,
            }

            # Training
            if len(self.buffer) < self.config.batch_size:
                print("  Buffer too small, skipping training.")
                iteration_record["training"] = {
                    "skipped": True,
                    "reason": "buffer_too_small",
                    "required_batch_size": self.config.batch_size,
                    "buffer_size": len(self.buffer),
                }
                self._append_history(iteration_record)
                continue

            t0 = time.time()
            self.network.train()
            metrics = self._train_epoch()
            train_seconds = time.time() - t0
            print(f"  Training: {self.config.training_steps_per_iteration} steps "
                  f"({train_seconds:.1f}s), "
                  f"policy_loss={metrics['policy_loss']:.4f}, "
                  f"value_loss={metrics['value_loss']:.4f}")
            iteration_record["training"] = {
                "skipped": False,
                "seconds": round(train_seconds, 4),
                "steps": self.config.training_steps_per_iteration,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
            }

            # Evaluation
            is_last = (iteration + 1 == self.config.num_iterations)
            evaluation_metrics = None
            if (iteration + 1) % self.config.eval_interval == 0 or is_last:
                evaluation_metrics = run_evaluation(self.config, self.network, self.device, iteration + 1)
            iteration_record["evaluation"] = evaluation_metrics

            # Checkpoint
            if (iteration + 1) % self.config.checkpoint_interval == 0 or is_last:
                self._save_checkpoint(iteration + 1, iteration_record)

            self._append_history(iteration_record)

    def _train_epoch(self):
        total_policy = 0.0
        total_value = 0.0
        steps = self.config.training_steps_per_iteration
        progress = make_progress(
            total=steps,
            desc="    training",
            unit="step",
            enabled=getattr(self.config, "show_progress_bars", True),
        )

        for step_idx in range(steps):
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
            if self.scheduler is not None:
                self.scheduler.step()

            total_policy += policy_loss.item()
            total_value += value_loss.item()

            progress.update(1)
            if step_idx == steps - 1 or (step_idx + 1) % max(1, steps // 25) == 0:
                progress.set_postfix(
                    policy=f"{total_policy / (step_idx + 1):.4f}",
                    value=f"{total_value / (step_idx + 1):.4f}",
                )

        progress.close()
        return {'policy_loss': total_policy / steps, 'value_loss': total_value / steps}

    def _save_checkpoint(self, iteration, iteration_record):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"iter_{iteration:04d}.pt"
        payload = {
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'buffer_size': len(self.buffer),
        }
        if self.scheduler is not None:
            payload['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(payload, path)
        metadata = {
            "schema_version": 1,
            "run_id": self.run_id,
            "saved_at": self._timestamp(),
            "checkpoint_path": self._relpath(path),
            "iteration": iteration,
            "config": asdict(self.config),
            "network": {
                "board_size": self.config.board_size,
                "num_channels": self.config.num_channels,
                "num_res_blocks": self.config.num_res_blocks,
                "input_planes": self.config.input_planes,
            },
            "search": {
                "num_simulations": self.config.num_simulations,
                "gumbel_sample_size": self.config.gumbel_sample_size,
                "gumbel_sigma_visit_c": self.config.gumbel_sigma_visit_c,
                "gumbel_sigma_scale_c": self.config.gumbel_sigma_scale_c,
                "pb_c_base": self.config.pb_c_base,
                "pb_c_init": self.config.pb_c_init,
                "use_gumbel_noise": self.config.use_gumbel_noise,
                "select_action_by_count": self.config.select_action_by_count,
                "select_action_by_softmax_count": self.config.select_action_by_softmax_count,
                "eval_select_action_by_count": self.config.eval_select_action_by_count,
                "eval_select_action_by_softmax_count": self.config.eval_select_action_by_softmax_count,
            },
            "training_run": {
                "checkpoint_dir": self._relpath(self.checkpoint_dir),
                "buffer_size": len(self.buffer),
            },
            "iteration_record": iteration_record,
        }
        self._write_json(path.with_suffix(".json"), metadata)
        self._write_latest_pointers(path)
        print(f"  Checkpoint saved: {path}")

    def _estimate_self_play_positions(self):
        if self.self_play_position_estimate is not None:
            return int(round(self.self_play_position_estimate))
        return max(
            self.config.num_self_play_games_per_iteration,
            int(round(0.80 * self.config.num_self_play_games_per_iteration * self.config.board_size * self.config.board_size)),
        )

    def _update_self_play_position_estimate(self, positions):
        if self.self_play_position_estimate is None:
            self.self_play_position_estimate = float(positions)
            return
        self.self_play_position_estimate = 0.7 * self.self_play_position_estimate + 0.3 * float(positions)

    @staticmethod
    def _profile_record(profile):
        if profile is None:
            return None
        profiled_seconds = profile.profiled_seconds
        avg_batch = profile.evaluated_states / profile.eval_batches if profile.eval_batches > 0 else 0.0
        return {
            "search_seconds": round(profile.search_seconds, 6),
            "prepare_seconds": round(profile.prepare_seconds, 6),
            "forward_seconds": round(profile.forward_seconds, 6),
            "finish_seconds": round(profile.finish_seconds, 6),
            "finalize_seconds": round(profile.finalize_seconds, 6),
            "store_seconds": round(profile.store_seconds, 6),
            "profiled_seconds": round(profiled_seconds, 6),
            "root_prepares": profile.root_prepares,
            "leaf_prepares": profile.leaf_prepares,
            "eval_batches": profile.eval_batches,
            "evaluated_states": profile.evaluated_states,
            "avg_eval_batch": round(avg_batch, 4),
            "search_calls": profile.search_calls,
            "commits": profile.commits,
        }

    @staticmethod
    def _format_self_play_profile(profile, wall_seconds):
        components = [
            ("search", profile.search_seconds),
            ("prepare", profile.prepare_seconds),
            ("forward", profile.forward_seconds),
            ("finish", profile.finish_seconds),
            ("finalize", profile.finalize_seconds),
            ("store", profile.store_seconds),
        ]
        summary = []
        denom = wall_seconds if wall_seconds > 0 else profile.profiled_seconds
        for label, seconds in components:
            pct = (100.0 * seconds / denom) if denom > 0 else 0.0
            summary.append(f"{label}={seconds:.2f}s ({pct:.0f}%)")
        avg_batch = profile.evaluated_states / profile.eval_batches if profile.eval_batches > 0 else 0.0
        summary.append(
            f"batches={profile.eval_batches}, states={profile.evaluated_states}, avg_batch={avg_batch:.1f}"
        )
        return ", ".join(summary)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        # Support full checkpoint dicts and plain model state_dicts (from supervised pretraining)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            self.network.load_state_dict(ckpt['model_state_dict'])
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception:
                # Optimizer/scheduler may be missing or incompatible; skip restoring them
                print("  Warning: optimizer or scheduler state not fully restored from checkpoint.")
            iteration = ckpt.get('iteration', 0)
            print(f"  Loaded checkpoint: {path} (iteration {iteration})")
            return iteration
        else:
            # Assume ckpt is a bare state_dict for the model
            self.network.load_state_dict(ckpt)
            print(f"  Loaded model state_dict from {path} (no optimizer/scheduler). Resuming from iteration 0.")
            return 0

    def _resolve_run_dir(self, resume_path):
        if resume_path is not None:
            if resume_path.parent.parent == self.checkpoint_root:
                return resume_path.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = (
            f"{timestamp}_b{self.config.board_size}"
            f"_c{self.config.num_channels}_r{self.config.num_res_blocks}"
            f"_s{self.config.num_simulations}"
        )
        return self.checkpoint_root / slug

    def _build_run_metadata(self, cli_args, argv, resume_path, existing_metadata=None):
        created_at = self._timestamp()
        base = {
            "schema_version": 1,
            "project": "HexGumbel",
            "run_id": self.checkpoint_dir.name,
            "created_at": created_at,
            "updated_at": created_at,
            "checkpoint_dir": self._relpath(self.checkpoint_dir),
            "resume_from": self._relpath(resume_path) if resume_path else None,
            "argv": list(argv),
            "cli_args": cli_args,
            "config": asdict(self.config),
            "environment": {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
            },
        }
        if existing_metadata is not None:
            base["created_at"] = existing_metadata.get("created_at", created_at)
            if "last_iteration" in existing_metadata:
                base["last_iteration"] = existing_metadata["last_iteration"]
        return base

    def _append_history(self, iteration_record):
        if self.history_path is None:
            return
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(iteration_record, sort_keys=True) + "\n")
        if self.run_metadata is not None:
            self.run_metadata["updated_at"] = self._timestamp()
            self.run_metadata["last_iteration"] = iteration_record["iteration"]
            self._write_json(self.run_metadata_path, self.run_metadata)

    @staticmethod
    def _timestamp():
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def _relpath(self, path):
        if path is None:
            return None
        return os.path.relpath(str(path), str(self.project_dir.parent))

    @staticmethod
    def _write_json(path, payload):
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")

    def _write_latest_pointers(self, latest_checkpoint):
        latest_run_path = self.checkpoint_root / "latest_run.txt"
        latest_run_path.write_text(f"{self._relpath(self.checkpoint_dir)}\n", encoding="utf-8")
        if latest_checkpoint is not None and str(latest_checkpoint).endswith(".pt"):
            latest_ckpt_path = self.checkpoint_root / "latest_checkpoint.txt"
            latest_ckpt_path.write_text(f"{self._relpath(latest_checkpoint)}\n", encoding="utf-8")
