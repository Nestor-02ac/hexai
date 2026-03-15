"""ResNet with policy + value heads for Hex."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_board(board, current_player: int) -> torch.Tensor:
    """Encode board as 3 planes: my stones, opponent stones, to-play indicator."""
    size = board.size
    n = size * size
    opponent = 3 - current_player

    # Build flat array first, then reshape — avoids per-cell Python overhead
    cells = [board.get_cell(i) for i in range(n)]
    state = np.zeros((3, n), dtype=np.float32)
    for i, c in enumerate(cells):
        if c == current_player:
            state[0, i] = 1.0
        elif c == opponent:
            state[1, i] = 1.0
    if current_player == 1:
        state[2, :] = 1.0

    return torch.from_numpy(state.reshape(3, size, size))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class HexNet(nn.Module):
    def __init__(self, board_size, num_channels=64, num_res_blocks=5):
        super().__init__()
        self.board_size = board_size
        action_size = board_size * board_size

        # Stem
        self.conv_stem = nn.Conv2d(3, num_channels, 3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * action_size, action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(action_size, num_channels)
        self.value_fc2 = nn.Linear(num_channels, 1)

    def forward(self, x):
        """
        x: (batch, 3, N, N)
        Returns: policy_logits (batch, N*N), value (batch, 1)
        """
        s = F.relu(self.bn_stem(self.conv_stem(x)))
        s = self.res_blocks(s)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(s)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(s)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v
