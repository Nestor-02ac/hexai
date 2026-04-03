"""
Configuration for the Y Gumbel AlphaZero implementation.
"""

from dataclasses import dataclass
import torch


@dataclass
class GumbelZeroConfig:

    board_size: int = 5 

    num_channels: int = 64
    num_res_blocks: int = 5
    input_planes: int = 4  # [my stones, opp stones, black-to-move, white-to-move]

    # MCTS / Gumbel Zero
    num_simulations: int = 0  # 0 = auto: number of cells
    gumbel_sample_size: int = 0  # 0 = auto: num_simulations

    gumbel_sigma_visit_c: float = 50.0
    gumbel_sigma_scale_c: float = 1.0

    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25

    reward_discount: float = 1.0
    value_flipping_player: int = 2  # White

    use_gumbel_noise: bool = True

    # Action selection (training)
    select_action_by_count: bool = False
    select_action_by_softmax_count: bool = True
    select_action_softmax_temperature: float = 1.0
    select_action_value_threshold: float = 0.1

    # Action selection (evaluation)
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

    num_iterations: int = 100
    checkpoint_interval: int = 5

    eval_games: int = 20
    eval_interval: int = 5
    eval_mcts_simulations: int = 1000

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_self_play_workers: int = 1
    mcts_backend: str = "auto"  # auto | python | cython
    show_progress_bars: bool = True
    profile_self_play: bool = False
    seed: int = 42

    use_lr_scheduler: bool = True

    def __post_init__(self):
        num_cells = self.board_size * (self.board_size + 1) // 2

        if self.num_simulations == 0:
            self.num_simulations = num_cells

        if self.gumbel_sample_size == 0:
            self.gumbel_sample_size = self.num_simulations

        # Ensure sample size ≤ action space
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

        # Infrastructure checks
        if self.num_self_play_workers < 1:
            raise ValueError("num_self_play_workers must be >= 1")

        if self.mcts_backend not in {"auto", "python", "cython"}:
            raise ValueError("mcts_backend must be one of: auto, python, cython")

    # Action space (Y-specific)
    @property
    def action_space(self) -> int:
        """
        Number of valid moves in Y: N(N+1)/2
        """
        return self.board_size * (self.board_size + 1) // 2

    # Validation helper
    @staticmethod
    def _validate_selection_mode(select_by_count: bool, select_by_softmax_count: bool, label: str):
        if select_by_count == select_by_softmax_count:
            raise ValueError(
                f"{label} action selection must enable exactly one of "
                "select_action_by_count or select_action_by_softmax_count"
            )