from train import Trainer
from Models import TicTacToeNet
from TicTacToeEnv import TicTacToe


def test_train():
    env = TicTacToe()

    # MCTS parameters
    mcts_args = {'c_puct': 2.0, 'num_simulations': 20}

    train_args = {
        'lr': 1e-3,
        'weight_decay': 3e-4,
        'batch_size': 64,
        'mcts_temperature': 1.0,
        'num_iterations': 2,
        'num_self_play': 2,
        'num_epochs': 2,
        'num_eval_matches': 4,
        'eval_win_margin': 0.05
    }

    train_args.update(mcts_args)

    policy = TicTacToeNet(env.action_size, env.action_size)

    alphaZero = Trainer(env, train_args, policy)
    alphaZero.train()


if __name__ == "__main__":
    test_train()
