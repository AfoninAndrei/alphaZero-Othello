import numpy as np
from train import Trainer
from Models import TicTacToeNet
from MCTS_model import MCTS
from envs.tic_tac_toe import TicTacToe
from test_cases_TicTacToe import (test_occupied_moves_not_chosen,
                                  test_move_after_middle_x, test_defence,
                                  test_move_for_current_board)


def play_policy_vs_rollout(inference_fn_probs, args, policy_player):
    # we do not use MCTS here for the policy
    # if the policy was properly trained, it should give
    # reasonable performance w/o MCTS
    env = TicTacToe()
    # Create the rollout agent.
    mcts = MCTS(env, args, None, True)

    state = env.get_initial_state()  # Shape: (board_size, board_size)
    current_player = 1  # Let the policy play as player 1.

    while True:
        valid_moves = env.get_valid_moves(state, current_player)
        if valid_moves.sum() == 0:
            return "Draw"

        if current_player == policy_player:
            probs = inference_fn_probs(state, current_player)
            probs = probs * valid_moves
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = valid_moves / valid_moves.sum()
            action = np.argmax(probs)
        else:
            # Rollout agent picks randomly among valid moves.
            action_probs = mcts.policy_improve_step(state,
                                                    current_player,
                                                    temp=0.0)
            action = np.argmax(action_probs)

        # Update the rollout agent tree (for consistency).
        mcts.make_move(action)
        state = env.get_next_state(state, action, current_player)
        reward, done = env.get_value_and_terminated(state, action,
                                                    current_player)
        if done:
            if reward == 1:
                # The player who just moved is the winner.
                return "Policy" if current_player == policy_player else "Rollout"
            elif reward == -1:
                return "Rollout" if current_player == policy_player else "Policy"
            else:
                return "Draw"

        current_player = env.get_opponent(current_player)


def test_policy_vs_rollout_win_rate(policy):
    """
    Plays multiple matches of Policy vs Rollout,
    alternating the role of the trained policy (player 1 for even games, -1 for odd games).
    Asserts that the trained policy wins at least 50% of games.
    """
    mcts_args = {'c_puct': 1.0, 'num_simulations': 200}
    # Merge MCTS args into test args.
    args = mcts_args.copy()

    num_matches = 100
    policy_wins, rollout_wins = 0, 0
    for i in range(num_matches):
        # Alternate: even-indexed games, policy plays as 1; odd-indexed, policy plays as -1.
        policy_player = 1 if (i % 2 == 0) else -1
        result = play_policy_vs_rollout(policy, args, policy_player)
        if result == "Policy":
            policy_wins += 1
        if result == 'Rollout':
            rollout_wins += 1

    win_rate = policy_wins / num_matches
    rollout_rate = rollout_wins / num_matches
    print(
        f"Policy win rate: {win_rate*100:.1f}%  ({policy_wins}/{num_matches})")
    assert win_rate >= rollout_rate, f"Expected policy win rate to be higher, got {win_rate*100:.1f}%"


def test_train():
    env = TicTacToe()

    mcts_args = {'c_puct': 2.0, 'num_simulations': 100}

    train_args = {
        'lr': 5e-3,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'max_train_samples': 5000,
        'train_steps_per_iter': 5000,
        'mcts_temperature': 1.0,
        'num_iterations': 10,
        'num_self_play': 300,
        'eval_win_margin': 0.05,
        'num_eval_matches': 4
    }

    train_args.update(mcts_args)

    policy = TicTacToeNet(env.action_size, env.action_size)

    alphaZero = Trainer(env, train_args, policy)
    alphaZero.train()

    inference_fn_probs = lambda x, pl: policy.inference(x, pl)[0]
    inference_fn_argmax = lambda x, pl: np.argmax(inference_fn_probs(x, pl))

    test_occupied_moves_not_chosen(inference_fn_argmax)
    test_move_after_middle_x(inference_fn_probs)
    test_move_for_current_board(inference_fn_probs)
    test_defence(inference_fn_argmax)
    test_policy_vs_rollout_win_rate(inference_fn_probs)


if __name__ == "__main__":
    test_train()
