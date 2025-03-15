import numpy as np

from train import Trainer
from Models import TicTacToeNet
from envs.tic_tac_toe import TicTacToe
from test_cases_TicTacToe import (test_occupied_moves_not_chosen,
                                  test_move_after_middle_x, test_defence,
                                  test_move_for_current_board)


def test_train():
    env = TicTacToe()

    mcts_args = {'c_puct': 2.0, 'num_simulations': 100}

    train_args = {
        'lr': 5e-3,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'mcts_temperature': 1.0,
        'num_iterations': 10,
        'num_self_play': 300,
        'num_epochs': 5,
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


if __name__ == "__main__":
    test_train()
