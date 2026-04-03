"""
ResNet policy/value network for the Game of Y.

Adapted from Hex:
- Triangular board embedded in square tensor
- Invalid cells are masked at inference time
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from y_board import Player


def encode_board(board, current_player: int) -> torch.Tensor:
    """
    Encode Y board as (4, size, size) tensor.

    Planes:
      0: current player's stones
      1: opponent stones
      2: current player indicator (all 1 if BLACK)
      3: current player indicator (all 1 if WHITE)

    Only lower triangle is used; rest is zero-padded.
    """
    size = board.size
    opponent = 3 - current_player

    state = np.zeros((4, size, size), dtype=np.float32)

    for r in range(size):
        for c in range(r + 1):
            idx = board.rc_to_idx(r, c)
            cell = board.board[idx]

            if cell == current_player:
                state[0, r, c] = 1.0
            elif cell == opponent:
                state[1, r, c] = 1.0

    if current_player == Player.BLACK:
        state[2, :, :] = 1.0
    else:
        state[3, :, :] = 1.0

    return torch.from_numpy(state)


def get_valid_mask(board):
    """
    Return mask of valid cells (size x size).
    1 = valid, 0 = invalid.
    """
    size = board.size
    mask = torch.zeros((size, size), dtype=torch.float32)

    for r in range(size):
        for c in range(r + 1):
            idx = board.rc_to_idx(r, c)
            if board.board[idx] == 0:
                mask[r, c] = 1.0

    return mask.view(-1)  # flattened for policy


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

class YNet(nn.Module):
    def __init__(self, board_size, num_channels=64, num_res_blocks=5):
        super().__init__()

        self.board_size = board_size
        self.action_size = board_size * board_size  # includes invalid cells

        # Shared trunk
        self.conv_stem = nn.Conv2d(4, num_channels, 3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(num_channels)

        self.res_blocks = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.action_size, self.action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.action_size, num_channels)
        self.value_fc2 = nn.Linear(num_channels, 1)

    def forward(self, x):
        """
        Input: (batch, 4, size, size)
        Returns:
          policy_logits: (batch, size*size)
          value: (batch, 1)
        """
        s = F.relu(self.bn_stem(self.conv_stem(x)))
        s = self.res_blocks(s)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(s)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(s)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

# Helper 

def masked_policy(policy_logits, board):
    """
    Apply mask to policy logits:
    invalid or occupied cells → -inf
    """
    mask = get_valid_mask(board).to(policy_logits.device)

    # Avoid log(0) issues
    masked_logits = policy_logits.clone()
    masked_logits[mask == 0] = -1e9

    return masked_logits



