import numpy as np
from .game import Game

# taken from https://github.com/suragnair/alpha-zero-general/tree/master/othello


class Board:
    __slots__ = ['n', 'pieces']
    # Define directions as a NumPy array for clarity
    __directions = np.array(
        [[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]],
        dtype=np.int8)

    def __init__(self, n):
        self.n = n
        # Initialize board as a NumPy array of zeros
        self.pieces = np.zeros((n, n), dtype=np.int8)
        mid = n // 2
        # Standard initial configuration for Othello
        self.pieces[mid - 1, mid] = 1
        self.pieces[mid, mid - 1] = 1
        self.pieces[mid - 1, mid - 1] = -1
        self.pieces[mid, mid] = -1

    def count_diff(self, color):
        """Count the difference between pieces of the given color and its opponent."""
        # Vectorized count: count True as 1, False as 0.
        return int(
            np.sum(self.pieces == color) - np.sum(self.pieces == -color))

    def get_legal_moves(self, color):
        moves = set()
        # Find all positions where the piece matches the color.
        positions = np.argwhere(self.pieces == color)
        for pos in positions:
            new_moves = self.get_moves_for_square(tuple(pos))
            moves.update(new_moves)
        return list(moves)

    def has_legal_moves(self, color):
        positions = np.argwhere(self.pieces == color)
        for pos in positions:
            if self.get_moves_for_square(tuple(pos)):
                return True
        return False

    def get_moves_for_square(self, square):
        i, j = square
        color = self.pieces[i, j]
        if color == 0:
            return []  # No moves from an empty square
        moves = []
        for d in self.__directions:
            move = self._discover_move(square, d)
            if move is not None:
                moves.append(move)
        return moves

    def execute_move(self, move, color):
        flips = []
        # Gather flips from all directions.
        for d in self.__directions:
            flips_dir = self._get_flips(move, d, color)
            if flips_dir:
                flips.extend(flips_dir)
        if not flips:
            raise ValueError("Illegal move executed: no flips found.")
        # Place the new piece at the move location.
        self.pieces[move[0], move[1]] = color
        # Flip the opponent pieces.
        for i, j in flips:
            self.pieces[i, j] = color

    def _discover_move(self, origin, direction):
        n = self.n
        di, dj = direction
        i, j = origin[0] + di, origin[1] + dj
        color = self.pieces[origin[0], origin[1]]
        flips = []
        while 0 <= i < n and 0 <= j < n:
            cell = self.pieces[i, j]
            if cell == 0:
                return (i, j) if flips else None
            elif cell == color:
                return None
            elif cell == -color:
                flips.append((i, j))
            i += di
            j += dj
        return None

    def _get_flips(self, origin, direction, color):
        n = self.n
        di, dj = direction
        flips = []
        i, j = origin[0] + di, origin[1] + dj
        while 0 <= i < n and 0 <= j < n:
            cell = self.pieces[i, j]
            if cell == 0:
                return []
            if cell == -color:
                flips.append((i, j))
            elif cell == color:
                return flips if flips else []
            else:
                return []
            i += di
            j += dj
        return []


class OthelloGame(Game):
    square_content = {-1: "X", +0: "-", +1: "O"}

    @staticmethod
    def get_square_piece(piece):
        return OthelloGame.square_content[piece]

    def __init__(self, n):
        self.n = n
        self._state_size = self.n * self.n
        self._action_size = self._state_size + 1

    @property
    def action_size(self):
        return self._action_size

    @property
    def state_size(self):
        return self._state_size

    def get_initial_state(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def get_next_state(self, state, action, player):
        if action == self.n * self.n:
            return state  # "Pass" move
        b = Board(self.n)
        b.pieces = state.copy()
        move = (action // self.n, action % self.n)
        b.execute_move(move, player)
        return b.pieces.copy()

    def get_valid_moves(self, state, player):
        valid = np.zeros(self.action_size, dtype=np.uint8)
        b = Board(self.n)
        b.pieces = state.copy()
        legal_moves = b.get_legal_moves(player)
        if not legal_moves:
            valid[-1] = 1  # Mark the "pass" move as valid.
            return valid
        for i, j in legal_moves:
            valid[self.n * i + j] = 1
        return valid

    def get_value_and_terminated(self, state, action, player):
        b = Board(self.n)
        b.pieces = state.copy()
        if b.has_legal_moves(player) or b.has_legal_moves(-player):
            return 0, False
        diff = b.count_diff(player)
        if diff > 0:
            return 1, True
        elif diff < 0:
            return -1, True
        else:
            return 0, True

    def get_canonical_form(self, state, player):
        # return state if player==1, else return -state if player==-1
        return player * state

    def get_symmetries(self, state, pi):
        assert len(pi) == self.n**2 + 1
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        sym = []
        for i in range(1, 5):
            for flip in [True, False]:
                new_board = np.rot90(state, i)
                new_pi = np.rot90(pi_board, i)
                if flip:
                    new_board = np.fliplr(new_board)
                    new_pi = np.fliplr(new_pi)
                sym.append((new_board, list(new_pi.ravel()) + [pi[-1]]))
        return sym

    def string_representation(self, state):
        return state.tobytes()

    def string_representation_readable(self, state):
        return "".join(self.square_content[square] for row in state
                       for square in row)

    def get_score(self, state, player):
        b = Board(self.n)
        b.pieces = state.copy()
        return b.count_diff(player)

    def get_opponent(self, player):
        return -player

    @staticmethod
    def display(state):
        n = state.shape[0]
        header = "   " + " ".join(str(i) for i in range(n))
        print(header)
        print("-" * (len(header) + 2))
        for y in range(n):
            row_str = f"{y} | " + " ".join(OthelloGame.square_content[state[y,
                                                                            x]]
                                           for x in range(n)) + " |"
            print(row_str)
        print("-" * (len(header) + 2))
