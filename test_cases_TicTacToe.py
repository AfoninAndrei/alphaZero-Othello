import numpy as np


def test_occupied_moves_not_chosen(inference_fn_argmax):
    # inference_fn_argmax should return chosen move
    # test that the occupied moves do not have high probs
    board = np.array([[1, 0, -1], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
    # X |   | O
    # ---------
    # X | X | O
    # ---------
    # O |   |
    move = inference_fn_argmax(board, 1)
    assert move not in (0, 2), (
        f"Model predicted an illegal/occupied move: {move}")


def test_defence(inference_fn_argmax):
    # inference_fn_argmax should return chosen move
    # test that the occupied moves do not have high probs
    board = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    # X |   |
    # ---------
    #   | X |
    # ---------
    # O |   |
    move = inference_fn_argmax(board, -1)
    assert move == 8, (f"Model predicted a bad move: {move}")


def test_move_after_middle_x(inference_fn_probs):
    """
    Ensure that the policy prioritizes corners (0,2,6,8) when X is in the center.
    inference_fn_probs should return action probabilities.
    """
    board = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                     dtype=np.float32)  # X in the middle
    #   |   |
    # ---------
    #   | X |
    # ---------
    #   |   |

    # Get probabilities from the inference function
    probs = inference_fn_probs(board, -1)
    top_4_indices = np.argsort(probs)[-4:]
    expected_top = {0, 2, 6, 8}  # Preferred corner moves

    assert top_4_indices[-1] in expected_top

    overlap = len(set(top_4_indices) & expected_top)  # Count common moves
    assert overlap >= 3, (f"Model did not prioritize corner moves. "
                          f"Expected {expected_top}, but got {top_4_indices}.")


def test_move_for_current_board(inference_fn_probs):
    """
    inference_fn_probs should return action probabilities.
    """
    board = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]],
                     dtype=np.float32)  # X in the middle
    # X |   |
    # ---------
    #   | O |
    # ---------
    #   |   | X

    # Get probabilities from the inference function
    probs = inference_fn_probs(board, -1)
    top_4_indices = np.argsort(probs)[-4:]
    expected_top = {1, 3, 5, 7}

    assert top_4_indices[-1] in expected_top

    overlap = len(set(top_4_indices) & expected_top)  # Count common moves
    assert overlap >= 3, (f"Model did not prioritize non-corner moves. "
                          f"Expected {expected_top}, but got {top_4_indices}.")
