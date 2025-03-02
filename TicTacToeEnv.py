import numpy as np


# taken from https://github.com/foersterrobert/AlphaZeroFromScratch
class TicTacToe:

    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]

        row_sum = np.sum(state[row, :])
        col_sum = np.sum(state[:, column])
        diag_sum = np.sum(np.diag(state))
        anti_sum = np.sum(np.diag(np.flipud(state)))

        size = self.row_count  # 3
        # If player is +1 or -1, the sum for a winning row (or col, diag) is either +3 or -3
        if (row_sum == player * size or col_sum == player * size
                or diag_sum == player * size or anti_sum == player * size):
            return True
        return False

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player
