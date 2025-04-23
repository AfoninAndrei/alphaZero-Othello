import numpy as np
import pytest

# Import the classes to be tested. Adjust the module path as needed.
from .othello import Board, _BitBoard


def numpy_board_from_bitboard(bb: _BitBoard) -> np.ndarray:
    """Convert bitboard to numpy array for direct comparison."""
    return bb.to_numpy()


def bitboard_moves_to_coords(mask: np.uint64, bb: _BitBoard):
    """Convert a bitmask of legal moves to a sorted list of (row, col) coordinates."""
    moves = []
    m = int(mask)
    while m:
        # Extract least significant bit index
        lsb_index = bb._lsb(np.uint64(m))
        moves.append((lsb_index // 8, lsb_index % 8))
        m &= m - 1  # Clear LSB
    return sorted(moves)


def test_initial_position_equivalence():
    board = Board(8)
    bb = _BitBoard()
    # Initial piece configuration should match
    np.testing.assert_array_equal(board.pieces, numpy_board_from_bitboard(bb))


def test_count_diff_and_score_initial():
    board = Board(8)
    bb = _BitBoard()
    # Difference for black should match bitboard score
    assert board.count_diff(1) == bb.score()
    # Difference for white should be negative of black's
    assert board.count_diff(-1) == -bb.score()


def test_legal_moves_initial():
    board = Board(8)
    bb = _BitBoard()
    # Board legal moves for black
    board_moves = sorted(board.get_legal_moves(1))
    # Bitboard valid moves mask
    bb_mask = bb.valid_mask()
    bb_moves = bitboard_moves_to_coords(bb_mask, bb)
    assert board_moves == bb_moves


def test_execute_move_equivalence():
    board = Board(8)
    bb = _BitBoard()

    # Play one opening move for black
    current_player = 1
    board_move = sorted(board.get_legal_moves(current_player))[0]
    action_black = board_move[0] * 8 + board_move[1]
    board.execute_move(board_move, current_player)
    bb.make_move(action_black)
    # After an odd number of moves, bitboard perspective flips to white-to-move,
    # so invert perspective to match static board orientation (black=1)
    np.testing.assert_array_equal(board.pieces, numpy_board_from_bitboard(bb))

    # Play one reply move for white
    current_player = -1
    board_move_white = sorted(board.get_legal_moves(current_player))[0]
    action_white = board_move_white[0] * 8 + board_move_white[1]
    board.execute_move(board_move_white, current_player)
    bb.make_move(action_white)
    # After two moves, perspective returns to black-to-move
    np.testing.assert_array_equal(board.pieces, numpy_board_from_bitboard(bb))


def test_board_execute_illegal_move_raises():
    board = Board(8)
    # Attempting to execute an illegal move should raise
    with pytest.raises(ValueError):
        board.execute_move((0, 0), 1)


def test_execute_move_equivalence():
    board = Board(8)
    bb = _BitBoard()

    current_player = 1

    for _ in range(5):  # Try 5 sequential moves alternating players
        current_player = 1 if board.count_diff(1) <= board.count_diff(
            -1) else -1
        move = sorted(board.get_legal_moves(current_player))[0]
        action = move[0] * 8 + move[1]
        board.execute_move(move, current_player)
        bb.make_move(action)

        current_player = -current_player

        # ✅ FIX: After move, _BitBoard flips, so use opponent's perspective
        np.testing.assert_array_equal(board.pieces,
                                      numpy_board_from_bitboard(bb))


def test_bitboard_no_valid_moves_pass():
    bb = _BitBoard()
    # Construct a full board manually
    bb.black = np.uint64(0xFFFFFFFFFFFFFFFF)
    bb.white = np.uint64(0)
    assert bb.valid_mask() == 0


def test_bitboard_score_full_black():
    bb = _BitBoard()
    bb.black = np.uint64(0xFFFFFFFFFFFFFFFF)
    bb.white = np.uint64(0)
    assert bb.score() == 64


def test_bitboard_score_full_white():
    bb = _BitBoard()
    bb.black = np.uint64(0)
    bb.white = np.uint64(0xFFFFFFFFFFFFFFFF)
    assert bb.score() == -64


def test_full_game_equivalence():
    board = Board(8)
    bb = _BitBoard()

    current_player = 1  # Black starts
    move_history = []

    # Run game until no player has legal moves
    passes = 0
    while passes < 2:
        legal_moves = board.get_legal_moves(current_player)
        if legal_moves:
            passes = 0
            move = sorted(legal_moves)[0]
            action = move[0] * 8 + move[1]

            # Apply to both board types
            board.execute_move(move, current_player)
            bb.make_move(action)
            move_history.append((current_player, move))

            # Compare states
            np.testing.assert_array_equal(
                board.pieces,
                numpy_board_from_bitboard(bb),
                err_msg=
                f"Mismatch after move {len(move_history)} by player {current_player} at {move}"
            )
        else:
            passes += 1
            bb.make_move(64)  # pass

        current_player = -current_player

    # Final score comparison
    board_score = board.count_diff(1)
    bitboard_score = bb.score()
    assert board_score == bitboard_score, (
        f"Final score mismatch: Board={board_score}, BitBoard={bitboard_score}"
    )

    print(f"✅ Full game equivalence test passed in {len(move_history)} moves")
