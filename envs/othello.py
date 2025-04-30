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


_MASKS = np.array(
    [
        np.uint64(0x7F7F7F7F7F7F7F7F),  # right
        np.uint64(0x007F7F7F7F7F7F7F),  # down-right
        np.uint64(0xFFFFFFFFFFFFFFFF),  # down
        np.uint64(0x00FEFEFEFEFEFEFE),  # down-left
        np.uint64(0xFEFEFEFEFEFEFEFE),  # left
        np.uint64(0xFEFEFEFEFEFEFE00),  # up-left
        np.uint64(0xFFFFFFFFFFFFFFFF),  # up
        np.uint64(0x7F7F7F7F7F7F7F00),  # up-right
    ],
    dtype=np.uint64)

_LSHIFTS = [0, 0, 0, 0, 1, 9, 8, 7]
_RSHIFTS = [1, 9, 8, 7, 0, 0, 0, 0]


class _BitBoard:
    """Internal 8×8 Othello bitboard.

    `black` always denotes the side to move; `white` is the opponent.
    """
    __slots__ = ("black", "white", "perspective")

    def __init__(self):
        # Starting position (black to move)
        self.black = np.uint64(0x0000000810000000)  # d5, e4
        self.white = np.uint64(0x0000001008000000)  # e5, d4
        self.perspective = 1

    @staticmethod
    def _lsb(mask: np.uint64) -> int:
        m = int(mask)
        return (m & -m).bit_length() - 1

    @staticmethod
    def _shift(disks: np.uint64, dir: int) -> np.uint64:
        """
        Bit-shift in direction dir (0–7), masking wrap-around exactly like C.
        """
        if dir < 4:
            return (disks >> _RSHIFTS[dir]) & _MASKS[dir]
        else:
            return (disks << _LSHIFTS[dir]) & _MASKS[dir]

    def _legal_moves(self, own: np.uint64, opp: np.uint64) -> np.uint64:
        """Return bitmask of all legal moves for `own` against `opp`."""
        empty = ~(own | opp)
        moves = np.uint64(0)
        for d in range(8):
            x = self._shift(own, d) & opp
            for _ in range(5):
                x |= self._shift(x, d) & opp
            moves |= self._shift(x, d) & empty
        return moves

    def valid_mask(self) -> np.uint64:
        return self._legal_moves(self.black, self.white)

    def make_move(self, action: int):
        """
        Play `action` (0–63) for the side-to-move (in `black`);
        use 64 to pass. Maintains `black` as the next side to move.
        """
        # pass move
        if action == 64:
            self.black, self.white = self.white, self.black
            self.perspective *= -1
            return

        new = np.uint64(1) << np.uint64(action)
        # include the new disk for bounding detection
        my = self.black | new
        opp = self.white
        captured = np.uint64(0)

        for d in range(8):
            x = self._shift(new, d) & opp
            for _ in range(5):
                x |= self._shift(x, d) & opp
            if self._shift(x, d) & my:
                captured |= x

        # apply move and flips
        self.black = my ^ captured
        self.white = opp ^ captured
        # switch side-to-move
        self.black, self.white = self.white, self.black
        self.perspective *= -1

    def to_numpy(self) -> np.ndarray:
        black, white = ((self.black, self.white) if self.perspective == 1 else
                        (self.white, self.black))
        board = np.zeros((8, 8), dtype=np.int8)
        for colour, mask in ((1, black), (-1, white)):
            m = mask
            while m:
                sq = self._lsb(m)
                board[sq // 8, sq % 8] = colour
                m &= m - 1
        return board

    def score(self) -> int:
        """
        Final score matching Board calculation:
        board.count_diff(1) - board.count_diff(-1) = 2*(black - white)
        """
        diff = int(int(self.black).bit_count() - int(self.white).bit_count())
        return diff


# helper popcount (used by any other code)
popcount = np.vectorize(lambda x: int(int(x).bit_count()), otypes=[int])


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

    def get_score(self, state, player):
        b = Board(self.n)
        b.pieces = state.copy()
        return b.count_diff(player)

    def get_opponent(self, player):
        return -player


class OthelloGameNew(Game):
    """
    Same public interface as the original Array‑based version but backed
    by the lock‑free 64‑bit bitboard core.
    """
    square_content = {-1: "X", 0: "-", 1: "O"}

    @staticmethod
    def get_square_piece(piece: int) -> str:
        return OthelloGameNew.square_content[piece]

    def __init__(self, n: int):
        # The bitboard works only for 8×8; keep the assert so errors are loud
        assert n == 8, "Bitboard engine supports only standard 8×8 Othello"
        self.n = n
        self._state_size = n * n
        self._action_size = self._state_size + 1  # +1 for the 'pass' move

    # -------- public properties (unchanged) ---------------------------
    @property
    def action_size(self):  # noqa: D401
        return self._action_size

    @property
    def state_size(self):  # noqa: D401
        return self._state_size

    def _idx_to_bit(idx: int) -> int:
        """
        Row‑major array index 0..63  →  bit index 0..63 used by _BitBoard.

        Array coordinates:
            row 0 = top, col 0 = left
        Bitboard coordinates (as used by the C core):
            bit 0 = A1 (bottom‑left), bit 63 = H8 (top‑right).

        We therefore need to flip *both* the vertical **and** horizontal
        axes when mapping.
        """
        row, col = divmod(idx, 8)
        return (7 - row) * 8 + (7 - col)  # ← flip row and col

    @staticmethod
    def _bit_to_idx(bit: int) -> int:
        """Inverse of `_idx_to_bit`."""
        row, col = divmod(bit, 8)
        return (7 - row) * 8 + (7 - col)

    @staticmethod
    def _np_to_bitboards(state: np.ndarray, player: int):
        flat = state.ravel()
        black = np.uint64(0)
        white = np.uint64(0)
        for idx, val in enumerate(flat):
            if val == player:
                # map row*8+col → bitboard index
                bb_idx = OthelloGameNew._idx_to_bit(idx)
                black |= np.uint64(1) << np.uint64(bb_idx)
            elif val == -player:
                bb_idx = OthelloGameNew._idx_to_bit(idx)
                white |= np.uint64(1) << np.uint64(bb_idx)
        return black, white

    @staticmethod
    def _bitboards_to_np(black: np.uint64, white: np.uint64) -> np.ndarray:
        board = np.zeros(64, np.int8)
        b = int(black)
        while b:
            sq = (b & -b).bit_length() - 1
            idx = OthelloGameNew._bit_to_idx(sq)
            board[idx] = 1
            b &= b - 1
        w = int(white)
        while w:
            sq = (w & -w).bit_length() - 1
            idx = OthelloGameNew._bit_to_idx(sq)
            board[idx] = -1
            w &= w - 1
        return board.reshape(8, 8)

    def get_initial_state(self):
        bb = _BitBoard()
        return self._bitboards_to_np(bb.black, bb.white)

    def get_valid_moves(self, state, player: int):
        black, white = self._np_to_bitboards(state, player)
        bb = _BitBoard()
        bb.black, bb.white = black, white

        mask = bb.valid_mask()
        valid = np.zeros(self._action_size, np.uint8)
        if mask == 0:
            valid[-1] = 1  # only pass
            return valid

        m = int(mask)
        while m:
            lsb = (m & -m).bit_length() - 1
            act_idx = OthelloGameNew._bit_to_idx(lsb)
            valid[act_idx] = 1
            m &= m - 1
        return valid

    def get_next_state(self, state, action: int, player: int):
        # pass
        if action == self._state_size:
            return state.copy()

        # illegal‐move guard
        valid = self.get_valid_moves(state, player)
        if valid[action] == 0:
            raise ValueError(f"Illegal move: {action}")

        # actual move
        black, white = self._np_to_bitboards(state, player)
        bb = _BitBoard()
        bb.black, bb.white = black, white
        # convert action (row*8+col) → bitboard index, then play
        bb.make_move(OthelloGameNew._idx_to_bit(action))

        if player == -1:
            return self._bitboards_to_np(bb.black, bb.white)
        else:
            return self._bitboards_to_np(bb.white, bb.black)

    def get_value_and_terminated(self, state, action, player):
        # Don’t check action legality here—just see if either side has moves
        # (we ignore `action` entirely, matching the old API).

        # If player still has any *non‑pass* moves, game continues
        if np.any(self.get_valid_moves(state, player)[:-1]):
            return 0, False

        # If opponent still has moves, game continues
        if np.any(self.get_valid_moves(state, -player)[:-1]):
            return 0, False

        # Otherwise game over — determine winner by piece count
        diff = np.sum(state == player) - np.sum(state == -player)
        if diff > 0:
            return 1, True
        elif diff < 0:
            return -1, True
        else:
            return 0, True

    def get_score(self, state, player):
        return int(np.sum(state == player) - np.sum(state == -player))

    def get_opponent(self, player):
        return -player


def get_random_symmetry(state: np.ndarray, pi: np.ndarray):
    """
    Apply one random dihedral–8 symmetry and return
    (state[C, n, n], pi[n*n+1]).
    """
    n = state.shape[-1]
    k = np.random.randint(4)
    flip = np.random.rand() < 0.5

    s = np.rot90(state, k, axes=(-2, -1))
    pb = np.rot90(pi[:-1].reshape(n, n), k)

    if flip:
        s = np.fliplr(s)
        pb = np.fliplr(pb)

    # --- add channel dim if state was 2-D ----------------------------------
    if s.ndim == 2:  # (n, n)  →  (1, n, n)
        s = s[None, :, :]

    # --- make both arrays positive-stride & contiguous ---------------------
    s = np.ascontiguousarray(s, dtype=np.float32)
    pi_out = np.concatenate([pb.ravel(), pi[-1:]]).astype(np.float32,
                                                          copy=False)

    return s, pi_out
