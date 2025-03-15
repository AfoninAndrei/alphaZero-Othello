import numpy as np
from .game import Game


# taken from https://github.com/foersterrobert/AlphaZeroFromScratch
class TicTacToe(Game):

    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self._action_size = self.row_count * self.column_count
        self._state_size = self._action_size

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    @property
    def action_size(self):
        return self._action_size

    @property
    def state_size(self):
        return self._state_size

    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state

    def get_valid_moves(self, state, player):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        """
        Check if the move at the given action produces a win.
        Returns:
            +1 or -1 if that player wins,
            0 if no win is detected.
        """
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]

        row_sum = np.sum(state[row, :])
        col_sum = np.sum(state[:, column])
        diag_sum = np.sum(np.diag(state))
        anti_sum = np.sum(np.diag(np.flipud(state)))

        size = self.row_count  # typically 3 in TicTacToe
        # If the sum of any row, column, or diagonal equals player * size,
        # then that player wins.
        if (row_sum == player * size or col_sum == player * size
                or diag_sum == player * size or anti_sum == player * size):
            return player
        return 0

    def get_value_and_terminated(self, state, action, current_player):
        # Check if the most recent move produced a win.
        winner = self.check_win(state, action)
        if winner != 0:
            # return reward from the current_player perspective
            return winner * current_player, True

        # If there are no valid moves left for the current player, the game is over.
        if np.sum(self.get_valid_moves(state, current_player)) == 0:
            return 0, True

        return 0, False

    def get_opponent(self, player):
        return -player
