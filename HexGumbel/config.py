"""Configuration for the Hex Gumbel AlphaZero implementation."""

from dataclasses import dataclass

import torch


@dataclass
class GumbelZeroConfig:
    # Board
    board_size: int = 7

    # Neural network
    num_channels: int = 64
    num_res_blocks: int = 5
    input_planes: int = 4  # [my stones, opp stones, black-to-move, white-to-move]

    # MCTS / Gumbel Zero
    num_simulations: int = 0  # 0 = auto: board_size^2
    gumbel_sample_size: int = 0  # 0 = auto: num_simulations
    gumbel_sigma_visit_c: float = 50.0
    gumbel_sigma_scale_c: float = 1.0
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    reward_discount: float = 1.0
    value_flipping_player: int = 2  # White
    use_gumbel_noise: bool = True
    select_action_by_count: bool = False
    select_action_by_softmax_count: bool = True
    select_action_softmax_temperature: float = 1.0
    select_action_value_threshold: float = 0.1
    eval_select_action_by_count: bool = True
    eval_select_action_by_softmax_count: bool = False
    eval_select_action_softmax_temperature: float = 1.0
    eval_select_action_value_threshold: float = 0.1

    # Self-play
    num_self_play_games_per_iteration: int = 100

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

    def __post_init__(self):
        if self.num_simulations == 0:
            self.num_simulations = self.board_size * self.board_size
        if self.gumbel_sample_size == 0:
            self.gumbel_sample_size = self.num_simulations
        self.gumbel_sample_size = min(self.gumbel_sample_size, self.action_space)
        self._validate_selection_mode(
            self.select_action_by_count,
            self.select_action_by_softmax_count,
            "training",
        )
        self._validate_selection_mode(
            self.eval_select_action_by_count,
            self.eval_select_action_by_softmax_count,
            "evaluation",
        )

    @property
    def action_space(self) -> int:
        return self.board_size * self.board_size

    @staticmethod
    def _validate_selection_mode(select_by_count: bool, select_by_softmax_count: bool, label: str):
        if select_by_count == select_by_softmax_count:
            raise ValueError(
                f"{label} action selection must enable exactly one of "
                "select_action_by_count or select_action_by_softmax_count"
            )
