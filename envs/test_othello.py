import unittest
import numpy as np
from othello import Board, OthelloGame


class TestBoard(unittest.TestCase):

    def setUp(self):
        self.n = 4
        self.board = Board(self.n)

    def test_initial_setup(self):
        # Expected initial board for a 4x4 board
        mid = self.n // 2
        expected = np.zeros((self.n, self.n), dtype=np.int8)
        expected[mid - 1, mid] = 1
        expected[mid, mid - 1] = 1
        expected[mid - 1, mid - 1] = -1
        expected[mid, mid] = -1
        np.testing.assert_array_equal(self.board.pieces, expected)

    def test_count_diff_initial(self):
        # At the initial board, each color has 2 pieces, so difference is zero.
        self.assertEqual(self.board.count_diff(1), 0)
        self.assertEqual(self.board.count_diff(-1), 0)

    def test_get_legal_moves(self):
        # Both players should have legal moves in the initial configuration.
        legal_moves_black = self.board.get_legal_moves(-1)
        legal_moves_white = self.board.get_legal_moves(1)
        self.assertTrue(len(legal_moves_black) > 0)
        self.assertTrue(len(legal_moves_white) > 0)

    def test_execute_move(self):
        # Pick a known legal move.
        # For a 4x4 board, one expected legal move for black (-1)
        # comes from the piece at (1,1) in the initial board.
        legal_moves_black = self.board.get_legal_moves(-1)
        self.assertIn((3, 1), legal_moves_black)
        # Execute the move (3,1) for black.
        self.board.execute_move((3, 1), -1)
        # After executing, the new piece should be set and the piece(s)
        # between should be flipped. For example, (2,1) should flip to -1.
        self.assertEqual(self.board.pieces[3, 1], -1)
        self.assertEqual(self.board.pieces[2, 1], -1)


class TestOthelloGame(unittest.TestCase):

    def setUp(self):
        self.n = 4
        self.game = OthelloGame(self.n)
        self.initial_state = self.game.get_initial_state()

    def test_initial_state(self):
        mid = self.n // 2
        expected = np.zeros((self.n, self.n), dtype=np.int8)
        expected[mid - 1, mid] = 1
        expected[mid, mid - 1] = 1
        expected[mid - 1, mid - 1] = -1
        expected[mid, mid] = -1
        np.testing.assert_array_equal(self.initial_state, expected)

    def test_get_valid_moves(self):
        valid_moves = self.game.get_valid_moves(self.initial_state, -1)
        self.assertEqual(valid_moves.shape[0], self.game.action_size)
        # Ensure there is at least one valid move.
        self.assertTrue(np.sum(valid_moves) > 0)

    def test_get_next_state_and_player_switch(self):
        # Get valid moves for black (-1) from the initial state.
        valid_moves = self.game.get_valid_moves(self.initial_state, -1)
        # Find a non-"pass" move (i.e. any move not at index action_size-1).
        indices = np.nonzero(valid_moves[:-1])[0]
        self.assertTrue(len(indices) > 0)
        action = indices[0]
        new_state = self.game.get_next_state(self.initial_state, -1, action)
        # The state should be updated.
        self.assertFalse(np.array_equal(new_state, self.initial_state))

    def test_get_value_and_terminated(self):
        # At the start, the game should not be terminated.
        value, terminated = self.game.get_value_and_terminated(
            self.initial_state, -1)
        self.assertFalse(terminated)
        self.assertEqual(value, 0)

    def test_symmetries(self):
        # Create a dummy policy vector (pi) with zeros.
        pi = [0] * (self.n * self.n) + [0]
        sym = self.game.get_symmetries(self.initial_state, pi)
        self.assertTrue(isinstance(sym, list))
        for board_sym, pi_sym in sym:
            self.assertEqual(board_sym.shape, (self.n, self.n))
            self.assertEqual(len(pi_sym), self.n * self.n + 1)


if __name__ == '__main__':
    unittest.main()
