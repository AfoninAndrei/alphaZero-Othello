from typing import Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class TicTacToeNet(nn.Module):

    def __init__(self, state_size, output_size):
        super().__init__()
        # Input dimension = 3*3 = 9 (board cells)
        # We'll embed each cell’s value (–1, 0, +1) directly.
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        value = torch.tanh(value)  # range in (–1, +1)

        return policy_logits, value

    def inference(self,
                  state: np.ndarray,
                  use_gpu: bool = False) -> Union[np.ndarray, float]:
        """
        Runs the model to get policy and value for a single TicTacToe state.

        :param model: a trained (or untrained) TicTacToeNet
        :param state: np.ndarray shape (3,3) with values in {–1, 0, +1}
        :param use_gpu: if True, move data to GPU before forward pass
        :return: (policy, value)
        policy: shape [9] numpy array of probabilities (not masked yet)
        value: float in [–1, +1], from this player's perspective
        """
        # Flatten the board to shape (9,)
        board_input = state.flatten().astype(np.float32)
        board_input = torch.from_numpy(board_input).unsqueeze(0)  # shape (1,9)

        if use_gpu:
            board_input = board_input.cuda()

        # Evaluate
        self.eval()
        with torch.no_grad():
            policy_logits, value = self(board_input)
            policy = self.softmax(policy_logits)
            # policy_logits -> shape (1,9)
            # value -> shape (1,1)

        # Convert to numpy
        policy = policy[0].cpu().numpy()  # shape (9,)
        value = value[0, 0].cpu().numpy().item()  # scalar

        return policy, value
