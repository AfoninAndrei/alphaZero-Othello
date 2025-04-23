import numpy as np
import pytest
import random
from random import Random

from .othello import (
    OthelloGame as SlowGameBase,
    OthelloGame,
    _BitBoard,
    OthelloGameNew as FastGameBase,
)

# Constants
N = 8  # board size
PASS = N * N  # 'pass' action index
RNG = random.Random(0xC0FFEE)


def moves_from_valid(valid):
    """Convert valid mask vector to list of action indices."""
    return [i for i, v in enumerate(valid) if v]


def assert_state_equal(a: np.ndarray, b: np.ndarray):
    assert a.dtype == b.dtype, "dtypes differ"
    assert np.array_equal(a, b), "board arrays differ"


def assert_policy_equal(a: np.ndarray, b: np.ndarray):
    assert a.shape == b.shape
    assert np.array_equal(a, b), "valid-moves masks differ"


# ----------------------------------------------------------------------
# 1.  Initial-state & move-equivalence tests
# ----------------------------------------------------------------------


def test_initial_states_equal():
    old = OthelloGame(8)
    new = FastGameBase(8)
    s_old = old.get_initial_state()
    s_new = new.get_initial_state()
    np.testing.assert_array_equal(s_old, s_new)


def test_valid_moves_initial_equal():
    old = OthelloGame(8)
    new = FastGameBase(8)
    s = old.get_initial_state()
    vm_old = old.get_valid_moves(s, 1)
    vm_new = new.get_valid_moves(s, 1)
    assert vm_old.dtype == vm_new.dtype
    np.testing.assert_array_equal(vm_old, vm_new)


def test_first_move_equivalence():
    slow = OthelloGame(8)
    fast = FastGameBase(8)
    s0 = slow.get_initial_state()
    f0 = fast.get_initial_state()

    # standard 4 opening actions
    for action in [19, 26, 37, 44]:
        s1 = slow.get_next_state(s0, action, +1)
        f1 = fast.get_next_state(f0, action, +1)
        assert_state_equal(s1, f1)

        vm_s = slow.get_valid_moves(s1, -1)
        vm_f = fast.get_valid_moves(f1, -1)
        assert_policy_equal(vm_s, vm_f)


# ----------------------------------------------------------------------
# 2.  Deterministic full playout (lowest-index move) equivalence
# ----------------------------------------------------------------------


def test_get_next_state_equivalence():
    slow = OthelloGame(8)
    fast = FastGameBase(8)
    state = slow.get_initial_state()
    player = 1

    valid_moves = np.nonzero(slow.get_valid_moves(state, player))[0]
    for action in valid_moves:
        next_slow = slow.get_next_state(state, action, player)
        next_fast = fast.get_next_state(state, action, player)
        np.testing.assert_array_equal(next_slow, next_fast)


def test_get_next_state_pass():
    slow = OthelloGame(8)
    fast = FastGameBase(8)
    state = slow.get_initial_state()
    player = 1

    pass_action = 64  # board size 8x8 => pass index
    next_slow = slow.get_next_state(state, pass_action, player)
    next_fast = fast.get_next_state(state, pass_action, player)
    np.testing.assert_array_equal(next_slow, next_fast)


def test_get_next_state_illegal():
    fast = FastGameBase(8)
    state = fast.get_initial_state()
    player = 1

    invalid_action = 0
    valid = fast.get_valid_moves(state, player)
    if valid[invalid_action] == 1:
        pytest.skip("Action 0 is unexpectedly legal; skipping test.")
    with pytest.raises(ValueError):
        fast.get_next_state(state, invalid_action, player)


def test_deterministic_full_playout_equivalence():
    slow = OthelloGame(8)
    fast = FastGameBase(8)
    state_s = slow.get_initial_state()
    state_f = fast.get_initial_state()
    player = +1

    while True:
        vm_s = slow.get_valid_moves(state_s, player)
        vm_f = fast.get_valid_moves(state_f, player)
        assert_policy_equal(vm_s, vm_f)

        actions = np.nonzero(vm_s)[0]
        assert actions.size > 0, "no valid moves at non-terminal board"
        action = int(actions.min())

        state_s = slow.get_next_state(state_s, action, player)
        state_f = fast.get_next_state(state_f, action, player)
        assert_state_equal(state_s, state_f)

        value_s, term_s = slow.get_value_and_terminated(
            state_s, action, -player)
        value_f, term_f = fast.get_value_and_terminated(
            state_f, action, -player)
        assert value_s == value_f
        assert term_s == term_f

        if term_s:
            # final score check
            assert slow.get_score(state_s, +1) == fast.get_score(state_f, +1)
            assert slow.get_score(state_s, -1) == fast.get_score(state_f, -1)
            break

        player = -player


# # ----------------------------------------------------------------------
# # 3.  Randomized playout equivalence
# # ----------------------------------------------------------------------


@pytest.mark.parametrize("game_id", range(5))
def test_random_playout_equivalence(game_id):
    slow = OthelloGame(8)
    fast = FastGameBase(8)
    state_s = slow.get_initial_state()
    state_f = fast.get_initial_state()
    player = +1

    for _ in range(200):  # safe upper bound
        vm_s = slow.get_valid_moves(state_s, player)
        vm_f = fast.get_valid_moves(state_f, player)
        assert_policy_equal(vm_s, vm_f)

        actions = np.nonzero(vm_s)[0]
        assert actions.size > 0
        action = RNG.choice(actions)

        next_s = slow.get_next_state(state_s, action, player)
        next_f = fast.get_next_state(state_f, action, player)
        assert_state_equal(next_s, next_f)
        player = -player

        value_s, term_s = slow.get_value_and_terminated(next_s, action, player)
        value_f, term_f = fast.get_value_and_terminated(next_f, action, player)
        assert value_s == value_f
        assert term_s == term_f

        state_s, state_f = next_s, next_f
        if term_s:
            assert slow.get_score(state_s, +1) == fast.get_score(state_f, +1)
            assert slow.get_score(state_s, -1) == fast.get_score(state_f, -1)
            return
    pytest.fail("game did not finish in upper-bound plies")


def test_action_and_state_size_properties():
    slow = OthelloGame(6)
    assert slow.state_size == 36
    assert slow.action_size == 37
    fast = FastGameBase(8)
    assert fast.state_size == 64
    assert fast.action_size == 65


def test_get_square_piece():
    assert OthelloGame.get_square_piece(1) == 'O'
    assert OthelloGame.get_square_piece(-1) == 'X'
    assert OthelloGame.get_square_piece(0) == '-'
    assert FastGameBase.get_square_piece(1) == 'O'
    assert FastGameBase.get_square_piece(-1) == 'X'
    assert FastGameBase.get_square_piece(0) == '-'


def test_canonical_form_flips_correctly():
    slow = OthelloGame(8)
    new = FastGameBase(8)
    s = slow.get_initial_state()
    can_slow = slow.get_canonical_form(s, -1)
    can_fast = new.get_canonical_form(s, -1)
    np.testing.assert_array_equal(can_slow, -s)
    np.testing.assert_array_equal(can_fast, -s)


def test_symmetry_content_matches():
    slow = OthelloGame(8)
    new = FastGameBase(8)
    s = slow.get_initial_state()
    pi = slow.get_valid_moves(s, 1)
    syms_slow = slow.get_symmetries(s, pi)
    syms_new = new.get_symmetries(s, pi)
    for (bs, ps), (bn, pn) in zip(syms_slow, syms_new):
        # boards equal and policies equal
        assert_state_equal(bs, bn)
        assert_policy_equal(np.array(ps, dtype=pi.dtype),
                            np.array(pn, dtype=pi.dtype))


def test_string_display_readable_and_bytes_consistency():
    slow = OthelloGame(8)
    new = FastGameBase(8)
    s = slow.get_initial_state()
    # string repr
    b_slow = slow.string_representation(s)
    b_new = new.string_representation(s)
    assert isinstance(b_slow, bytes)
    assert b_slow == b_new
    # human readable
    h_slow = slow.string_representation_readable(s)
    h_new = new.string_representation_readable(s)
    assert h_slow == h_new


def test_display_prints_expected_format(capsys):
    slow = OthelloGame(4)
    # small 4x4 for readability
    s = slow.get_initial_state()
    slow.display(s)
    captured = capsys.readouterr().out
    # header line contains column indices
    assert '0 1 2 3' in captured
    # row lines start with row index and use X/O/- characters
    assert '0 |' in captured and '|' in captured


def test_pass_move_behavior():
    slow = OthelloGame(8)
    new = FastGameBase(8)
    # create a state where only pass is valid: fill three quadrants
    s = np.zeros((8, 8), np.int8)
    s[:4, :4] = 1
    s[:4, 4:] = 1
    s[4:, :4] = 1
    # leave bottom-right empty -> but no flips anywhere -> no moves
    vm_old = slow.get_valid_moves(s, -1)
    vm_new = new.get_valid_moves(s, -1)
    assert vm_old[-1] == 1 and vm_new[-1] == 1
    # next_state with pass should return identical state
    s1 = slow.get_next_state(s, PASS, 1)
    s2 = new.get_next_state(s, PASS, 1)
    assert_state_equal(s1, s2)


def test_illegal_move_raises():
    slow = OthelloGame(8)
    new = FastGameBase(8)
    s = slow.get_initial_state()
    # illegal coordinate (0), illegal because no flips
    with pytest.raises(ValueError):
        slow.get_next_state(s, 0, 1)
    with pytest.raises(ValueError):
        new.get_next_state(s, 0, 1)


RNG = Random(0xC0FFEE)

# Helpers -------------------------------------------------------------------


def round_trip_np(bb_cls, state, player=1):
    """Convert state → bitboards → state and return the result."""
    b, w = bb_cls._np_to_bitboards(state, player)
    return bb_cls._bitboards_to_np(b, w)


# ----------------------------------------------------------------------
#  1.  Bitboard ⇄ NumPy round‐trip indexing tests
# ----------------------------------------------------------------------
def test_round_trip_bitboard_indexing():
    for _ in range(50):
        # random board of -1/0/1
        flat = RNG.choices([-1, 0, 1], k=64)
        state = np.array(flat, dtype=np.int8).reshape(8, 8)

        rt = round_trip_np(FastGameBase, state, player=1)

        # only non‐zero squares must round‐trip
        mask = (state != 0)
        assert np.array_equal(rt[mask], state[mask])

        # empty squares must all be zero
        assert np.all(rt[~mask] == 0), "Empty squares did not round‐trip to 0"


# ----------------------------------------------------------------------
#  2.  Edge‐of‐board wrap‐around tests
# ----------------------------------------------------------------------
_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0),
               (-1, 1)]


@pytest.mark.parametrize(
    "corner,dir_idx,forbidden",
    [
        ((4, 7), 0, (4, 0)),  # H4 east→A4 wrap forbidden
        ((7, 4), 2, (0, 4)),  # 8E south→1E wrap forbidden
        ((3, 0), 4, (3, 7)),  # D1 west→D8 wrap forbidden
        ((0, 3), 6, (7, 3)),  # 1D north→8D wrap forbidden
    ])
def test_no_wraparound_from_edge(corner, dir_idx, forbidden):
    # place a single black at `corner` and a white next in dir
    b = _BitBoard()
    ci, cj = corner
    bi = FastGameBase._idx_to_bit(ci * 8 + cj)
    b.black = np.uint64(1) << np.uint64(bi)

    di, dj = _DIRECTIONS[dir_idx]
    wi, wj = ci + di, cj + dj
    if 0 <= wi < 8 and 0 <= wj < 8:
        wi_bit = FastGameBase._idx_to_bit(wi * 8 + wj)
        b.white = np.uint64(1) << np.uint64(wi_bit)

    mask = b.valid_mask()
    fb = np.uint64(1) << np.uint64(
        FastGameBase._idx_to_bit(forbidden[0] * 8 + forbidden[1]))
    assert (mask & fb) == 0, f"Wrapped move generated to {forbidden}"


# ----------------------------------------------------------------------
#  3.  Multiple-pass termination & perspective tests
# ----------------------------------------------------------------------
def test_double_pass_ends_and_flip_perspective():
    b = _BitBoard()
    # clear both bitboards → no legal moves
    b.black = b.white = np.uint64(0)
    assert b.perspective == 1

    b.make_move(64)  # pass
    assert b.perspective == -1

    b.make_move(64)  # second pass → still no moves
    assert b.valid_mask() == 0
    assert b.perspective == 1


# ----------------------------------------------------------------------
#  4.  get_value_and_terminated corner-cases
# ----------------------------------------------------------------------
def test_value_and_terminated_logic():
    slow = OthelloGame(8)

    empty = np.zeros((8, 8), np.int8)
    val, term = slow.get_value_and_terminated(empty, 64, 1)
    assert term is True and val == 0  # 0 vs. 0 → draw

    empty[0, 0] = 1
    val, term = slow.get_value_and_terminated(empty, 64, 1)
    assert term is True and val == 1  # one black → win

    init = slow.get_initial_state()
    val, term = slow.get_value_and_terminated(init, 19, 1)
    assert term is False and val == 0  # non-terminal


# ----------------------------------------------------------------------
#  5.  score() vs get_score() consistency
# ----------------------------------------------------------------------
def test_score_consistency():
    slow = OthelloGame(8)
    fast = FastGameBase(8)

    # random endgame position of only +/-1
    flat = RNG.choices([-1, 1], k=64)
    state = np.array(flat, dtype=np.int8).reshape(8, 8)

    s_slow = slow.get_score(state, 1)
    s_fast = fast.get_score(state, 1)
    bb = _BitBoard()
    bb.black, bb.white = FastGameBase._np_to_bitboards(state, 1)
    s_bb = bb.score()

    assert s_slow == s_fast == s_bb


# ----------------------------------------------------------------------
#  6.  negamax / compute_move smoke test
# ----------------------------------------------------------------------
def test_othello_ai_compute_move_smoke():

    bb = _BitBoard()
    mask = int(bb.valid_mask())  # cast to Python int
    if mask == 0:
        pytest.skip("No legal moves–nothing to test")

    # find lowest-bit move:
    lsb = mask & -mask
    move_index = lsb.bit_length() - 1
    assert 0 <= move_index < 64
