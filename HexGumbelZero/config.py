"""All hyperparameters in one place."""

from dataclasses import dataclass
import torch


@dataclass
class GumbelZeroConfig:
    # Board
    board_size: int = 7

    # Neural network
    num_channels: int = 64
    num_res_blocks: int = 5
    input_planes: int = 3  # [my_stones, opp_stones, to_play]

    # Gumbel MCTS
    num_simulations: int = 50
    max_considered_actions: int = 16
    maxvisit_init: int = 50
    value_scale: float = 0.1
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25

    # Self-play
    num_self_play_games_per_iteration: int = 100
    temperature: float = 1.0
    temperature_drop_move: int = 30

    # Training
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    training_steps_per_iteration: int = 200
    value_loss_weight: float = 1.0
    replay_buffer_capacity: int = 100_000

    # Outer loop
    num_iterations: int = 100
    checkpoint_interval: int = 5

    # Evaluation
    eval_games: int = 20
    eval_interval: int = 5
    eval_mcts_simulations: int = 1000

    # Infrastructure
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_self_play_workers: int = 1
    seed: int = 42

    @property
    def action_space(self) -> int:
        return self.board_size * self.board_size
