"""ResNet policy/value network for Hex."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hex_board import Player


def encode_board(board, current_player: int) -> torch.Tensor:
    """Encode the board with 4 feature planes and no spatial transpose."""
    size = board.size
    n = size * size
    opponent = 3 - current_player

    cells = [board.get_cell(i) for i in range(n)]
    state = np.zeros((4, n), dtype=np.float32)
    for i, cell in enumerate(cells):
        if cell == current_player:
            state[0, i] = 1.0
        elif cell == opponent:
            state[1, i] = 1.0

    if current_player == Player.BLACK:
        state[2, :] = 1.0
    else:
        state[3, :] = 1.0

    return torch.from_numpy(state.reshape(4, size, size))


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
        action_size = board_size * board_size

        self.conv_stem = nn.Conv2d(4, num_channels, 3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.Sequential(*[ResBlock(num_channels) for _ in range(num_res_blocks)])

        self.policy_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * action_size, action_size)

        self.value_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(action_size, num_channels)
        self.value_fc2 = nn.Linear(num_channels, 1)

    def forward(self, x):
        s = F.relu(self.bn_stem(self.conv_stem(x)))
        s = self.res_blocks(s)

        p = F.relu(self.policy_bn(self.policy_conv(s)))
        p = p.view(p.size(0), -1)
        policy_logit = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(s)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logit, value
