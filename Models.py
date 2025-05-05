from typing import Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Inference:

    def inference(self, state: np.ndarray,
                  current_player: int) -> Union[np.ndarray, float]:
        """
        Runs the model to get policy and value for a single state.
        """
        board_input = (current_player * state).astype(np.float32)
        board_input = torch.from_numpy(board_input).unsqueeze(0)

        device = next(self.parameters()).device
        board_input = board_input.to(device)
        # Evaluate
        self.eval()
        with torch.no_grad():
            policy_logits, value = self(board_input)
            policy = self.softmax(policy_logits)
        # Convert to numpy
        policy = policy[0].cpu().numpy()
        value = value[0, 0].cpu().numpy().item()  # scalar

        return policy, value


class TicTacToeNet(nn.Module, Inference):

    def __init__(self, state_size, output_size):
        super().__init__()
        # Input dimension = 3*3 = 9 (board cells)
        # We'll embed each cell’s value (–1, 0, +1) directly.
        self.action_size = output_size
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)

        # Policy head (outputs a logit for each action 0..8)
        self.policy_head = nn.Linear(64, output_size)

        # Value head (outputs a single scalar in –1..+1, so we often do a tanh)
        self.value_head = nn.Linear(64, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """
        :param x: a float tensor of shape [batch_size, 9], representing board states
                  (flattened 3x3). x can be –1, 0, or +1 in each cell.
        :return: (policy_logits, value)
          policy_logits shape: [batch_size, 9]
          value shape: [batch_size, 1]
        """
        # Basic MLP
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        value = torch.tanh(value)  # range in (–1, +1)

        return policy_logits, value


class ResidualBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return F.relu(out)


# The FastOthelloNet architecture adapted for fast convergence.
class FastOthelloNet(nn.Module, Inference):

    def __init__(self, board_size: int, action_size: int):
        """
        :param board_size: size of the board (e.g., 8 for an 8x8 board)
        :param action_size: number of actions (board_size*board_size + 1 for the "pass" move)
        """
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size

        # Initial convolution block with BatchNorm.
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU())

        # A residual block that helps gradients flow and accelerates learning.
        self.res_block = ResidualBlock(64)

        # An additional convolutional block to further extract features.
        self.conv_add = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU())

        # Compute the flattened dimension.
        flatten_dim = 64 * board_size * board_size

        # Policy head: maps flattened features to action logits.
        self.fc_policy = nn.Linear(flatten_dim, action_size)

        # Value head: uses an FC layer with increased capacity for faster convergence.
        self.fc_value1 = nn.Linear(flatten_dim, 64)
        self.fc_value2 = nn.Linear(64, 1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor of shape [batch_size, board_size, board_size] with values in {-1, 0, +1}.
        :return: tuple (policy_logits, value)
                 - policy_logits: Tensor of shape [batch_size, action_size]
                 - value: Tensor of shape [batch_size, 1] in the range (-1, +1)
        """

        if x.dim() == 3:
            x = x.unsqueeze(1)  # Shape: [batch, 1, board_size, board_size]

        # Pass through the initial conv and residual block.
        x = self.initial_conv(x)
        x = self.res_block(x)
        x = self.conv_add(x)

        # Flatten feature maps.
        x_flat = x.view(x.size(0), -1)

        # Policy head output.
        policy_logits = self.fc_policy(x_flat)

        # Value head output.
        value_intermediate = F.relu(self.fc_value1(x_flat))
        value = torch.tanh(self.fc_value2(value_intermediate))

        return policy_logits, value


class AlphaZeroNet(nn.Module, Inference):

    def __init__(self,
                 board_size: int,
                 action_size: int,
                 n_res_blocks: int = 5,
                 channels: int = 128):
        super().__init__()

        # -------- shared trunk --------
        self.conv0 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(channels)
        self.res = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(n_res_blocks)])

        # -------- policy head --------
        self.pol_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.pol_bn = nn.BatchNorm2d(2)
        self.pol_fc = nn.Linear(2 * board_size * board_size, action_size)

        # -------- value head ---------
        self.val_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(board_size * board_size, 256)
        self.val_fc2 = nn.Linear(256, 1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Shape: [batch, 1, board_size, board_size]

        x = F.relu(self.bn0(self.conv0(x)))
        x = self.res(x)

        # policy
        p = F.relu(self.pol_bn(self.pol_conv(x)))
        p = self.pol_fc(p.view(p.size(0), -1))

        # value
        v = F.relu(self.val_bn(self.val_conv(x)))
        v = F.relu(self.val_fc1(v.view(v.size(0), -1)))
        v = torch.tanh(self.val_fc2(v))

        return p, v
