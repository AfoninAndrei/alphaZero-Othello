import numpy as np
from envs.tic_tac_toe import TicTacToe
from MCTS_model import MCTS
from test_cases_TicTacToe import (test_occupied_moves_not_chosen, test_defence)


def test_mcts_for_forced_win():
    env = TicTacToe()

    # Pre-set a board where player=1 has two in a row on top
    # board layout:
    #  [ [1, 1, 0],
    #    [0, 0, 0],
    #    [0, 0, 0] ]
    state = env.get_initial_state()
    state[0, 0] = 1
    state[0, 1] = 1

    # The winning move is (0,2) => action index=2 ( row=0 * 3 + col=2 )
    args = {
        'c_puct': 1.0,
        'num_simulations':
        200  # Increase to ensure MCTS can discover the forced win
    }

    mcts = MCTS(env, args, None)
    # It's player=1's turn
    action_probs = mcts.policy_improve_step(state, init_player=1, temp=0.0)

    # The best move should be action=2 for an immediate win.
    # Let's see if MCTS strongly favors that move:
    best_action = np.argmax(action_probs)

    assert best_action == 2, "MCTS did not choose the winning move!"

    print("Test passed: MCTS found the forced win move at (0,2).")


def test_strategies():
    env = TicTacToe()
    args = {
        'c_puct': 1.0,
        'num_simulations':
        200  # Increase to ensure MCTS can discover the forced win
    }
    mcts = MCTS(env, args, None)
    inference_fn_probs = lambda x, pl: mcts.policy_improve_step(
        x, init_player=pl, temp=0.0)
    inference_fn_argmax = lambda x, pl: np.argmax(inference_fn_probs(x, pl))

    test_occupied_moves_not_chosen(inference_fn_argmax)
    # reset the tree
    mcts.root = None
    test_defence(inference_fn_argmax)


def play_mcts_vs_random(args):
    env = TicTacToe()
    mcts = MCTS(env, args, None)

    state = env.get_initial_state()
    # it is harder to win if you start second
    current_player = 1

    while True:
        valid_moves = env.get_valid_moves(state, current_player)
        if valid_moves.sum() == 0:
            # no moves => draw
            return "Draw"

        if current_player == -1:
            # MCTS picks action
            action_probs = mcts.policy_improve_step(state,
                                                    current_player,
                                                    temp=0.0)
            action = np.argmax(action_probs)
        else:
            # random picks action
            possible_actions = np.where(valid_moves == 1)[0]
            action = np.random.choice(possible_actions)

        mcts.make_move(action)
        state = env.get_next_state(state, action, current_player)
        reward, done = env.get_value_and_terminated(state, action,
                                                    current_player)
        if done:
            if reward == 1:
                # current player wins
                return "MCTS" if current_player == -1 else "Random"
            elif reward == -1:
                # current player loses; so opponent wins
                return "Random" if current_player == -1 else "MCTS"
            else:
                return "Draw"
        current_player = env.get_opponent(current_player)


def test_mcts_vs_random_win_rate():
    """
    Runs multiple matches. 
    Checks that MCTS wins at least a given threshold of the time.
    """
    args = {
        'c_puct': 1.0,
        'num_simulations': 200  # You can adjust if you want stronger MCTS
    }

    num_matches = 100  # number of games to play
    mcts_wins = 0

    for _ in range(num_matches):
        result = play_mcts_vs_random(args)
        if result == "MCTS":
            mcts_wins += 1

    # Calculate the win rate
    win_rate = mcts_wins / num_matches
    print(f"MCTS win rate: {win_rate * 100:.1f}%  ({mcts_wins}/{num_matches})")

    # Assert MCTS wins at least 50% of games (adjust threshold as desired)
    assert win_rate >= 0.50, f"Expected MCTS to win >=50%, got {win_rate * 100:.1f}%"


if __name__ == "__main__":
    test_mcts_for_forced_win()
    test_mcts_vs_random_win_rate()
    test_strategies()
