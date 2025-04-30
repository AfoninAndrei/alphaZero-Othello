import time
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import numpy as np
from multiprocessing import get_context

from eval import evaluate_models_parallel, _worker_init
from self_play_worker import one_self_play
from Models import FastOthelloNet
from envs.othello import get_random_symmetry

# TODO: why can't we beat the rollout 100% consistently?
# model is too weak? target is too noisy?
# probably since we still cannot beat the supervised model
# we need to adapt the noiseness of the gt

# TODO: to get the most out of the collected data
# and utilize the GPU more since it is not a bottleneck
# (training time ~12sec, data collection ~80sec) we can
# add augmentations to the training and increase number of epochs

# TODO: try as a learning target to use q + z, instead of z
# this should help to reduce the noise, also if we have duplicating data after data collection
# maybe it makes sense to filter it? Also maybe it makes sense to construct td-lambda target?

# TODO: add to the blog explanation on why we choose 100 simulations in the tree
# it feels like starting from this number we have good estimation of the win
# variance is lower

# Speed up story: Probably makes sense to remeasure everything
# 1. multiprocessing for self-play ~4x self-play speed up
# 2. multiprocessing for the evaluation ~4x eval speed up
# 3. Training on the mps ~2x speed up
# 4. Faster othello env ~25% speed up


class RandomSymmetryDataset(Dataset):

    def __init__(self, states, policies, values):
        self.states = states  # still plain numpy arrays / lists
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        s, p = get_random_symmetry(self.states[idx], self.policies[idx])
        v = self.values[idx]

        return (
            torch.from_numpy(s),  # now contiguous, float32
            torch.from_numpy(p),  # contiguous, float32
            torch.tensor(v, dtype=torch.float32))


class Trainer:

    def __init__(self, board_size, args, policy):
        self.board_size = board_size
        self.args = args
        self.policy = policy
        self.num_simulations = self.args['num_simulations']

        self.train_device = torch.device("mps")
        # important to copy, otherwise we update both models during training
        self.best_policy = copy.deepcopy(self.policy).eval()

        self.value_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.args['lr'],
            weight_decay=self.args['weight_decay'])

        self.current_training_data = []
        self.dataloader = None

        self.metrics_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'current_win_rate': [],
            'best_win_rate': [],
            'current_win_rate_rollout': [],
            'best_win_rate_rollout': []
        }

        self.self_play_history = {'winner': []}

    def clean_training_data(self):
        self.current_training_data = []

    def setup_dataloader(self):
        """
        Convert self.current_training_data (list of (state, policy, value))
        into a PyTorch DataLoader for training.
        """

        states = []
        policies = []
        values = []
        for (s, p, v) in self.current_training_data:
            states.append(s)  # 2D array: shape (board_size, board_size)
            policies.append(p)  # policy vector, shape (action_size,)
            values.append(v)  # scalar

        states = np.array(states, dtype=np.float32)
        policies = np.array(policies, dtype=np.float32)
        values = np.array(values, dtype=np.float32).reshape(-1, 1)
        # dataset = TensorDataset(torch.from_numpy(states),
        #                         torch.from_numpy(policies),
        #                         torch.from_numpy(values))
        dataset = RandomSymmetryDataset(states, policies, values)
        self.dataloader = DataLoader(dataset,
                                     batch_size=self.args['batch_size'],
                                     shuffle=True)

    def collect_self_play_games(self):
        """Run `num_self_play` games in parallel and extend
        `self.current_training_data`."""
        ctx = get_context("spawn")
        base_seed = np.random.randint(1_000_000)
        manager = ctx.Manager()
        # there is no need to clean this cache until we
        # find a new best model that does self-play
        shared_cache = manager.dict()
        with ctx.Pool(self.args["num_workers"],
                      initializer=_worker_init,
                      initargs=(base_seed, )) as pool:
            # Prepare a tuple of arg‑tuples so we can stream them
            work_items = [(self.board_size, self.args,
                           self.best_policy.state_dict(), shared_cache)
                          for _ in range(self.args["num_self_play"])]

            # chunksize=1 → each worker returns as soon as it finishes
            for traj in pool.imap_unordered(one_self_play,
                                            work_items,
                                            chunksize=1):
                self.current_training_data.extend(traj)

    def train_iters(self):
        self.policy.to(self.train_device)
        self.policy.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_batches = 0

        for (state, target_policy, target_value) in self.dataloader:
            # predict
            state = state.to(self.train_device)
            target_policy = target_policy.to(self.train_device)
            target_value = target_value.to(self.train_device)

            policy_logits, value = self.policy(state)
            # compute policy loss
            log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_loss = -torch.mean(
                torch.sum(target_policy * log_probs, dim=-1))
            # compute value loss
            value_loss = self.value_loss(value, target_value)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.cpu().item()
            total_value_loss += value_loss.cpu().item()
            total_batches += 1

        avg_policy_loss = total_policy_loss / total_batches
        avg_value_loss = total_value_loss / total_batches
        avg_total_loss = avg_policy_loss + avg_value_loss

        self.metrics_history['policy_loss'].append(avg_policy_loss)
        self.metrics_history['value_loss'].append(avg_value_loss)
        self.metrics_history['total_loss'].append(avg_total_loss)

        print(f"Policy Loss={self.metrics_history['policy_loss'][-1]:.4f}, "
              f"Value Loss={self.metrics_history['value_loss'][-1]:.4f}")

    def train(self):
        for _ in range(self.args['num_iterations']):

            # self-play
            start_time = time.time()
            self.collect_self_play_games()

            print("Collection Time taken", time.time() - start_time)

            print(f"Collected {len(self.current_training_data)} positions")
            # train
            self.setup_dataloader()

            start_time = time.time()
            for _ in range(self.args['num_epochs']):
                self.train_iters()
            print("Train Time taken", time.time() - start_time)

            start_time = time.time()
            self.eval()
            print("Eval Time taken", time.time() - start_time)

    def eval(self):
        self.policy.to("cpu")
        self.policy.eval()
        current_win_rate, best_win_rate = evaluate_models_parallel(
            self.board_size,
            self.args,
            self.policy.state_dict(),
            self.best_policy.state_dict(),
            n_matches=self.args['num_eval_matches'])

        print(
            f"Win Rate Current {current_win_rate*100:.1f}% vs Win rate Best: {best_win_rate*100:.1f}%"
        )

        self.metrics_history['current_win_rate'].append(current_win_rate)
        self.metrics_history['best_win_rate'].append(best_win_rate)

        # If better than the best model by a margin, accept new model as best
        if current_win_rate - self.args['eval_win_margin'] > best_win_rate:
            print("Replacing best model with current model")
            self.best_policy = copy.deepcopy(self.policy)
            torch.save(self.best_policy, 'othello_policy_RL.pt')

            self.clean_training_data()
            current_win_rate, best_win_rate = evaluate_models_parallel(
                self.board_size,
                self.args,
                self.policy.state_dict(),
                None,
                n_matches=30)
            self.metrics_history['current_win_rate_rollout'].append(
                current_win_rate)
            self.metrics_history['best_win_rate_rollout'].append(best_win_rate)
            print(
                f"Eval against rollout: Win Rate Current {current_win_rate*100:.1f}% vs Win rate Best: {best_win_rate*100:.1f}%"
            )


if __name__ == '__main__':
    mcts_args = {
        'c_puct': 2.0,
        'num_simulations': 100,
        'dirichlet_alpha': 0.1,
        'dirichlet_epsilon': 0.25
    }

    train_args = {
        'lr': 3e-4,
        'weight_decay': 3e-6,
        'batch_size': 64,
        'mcts_temperature': 1.0,
        'num_iterations': 200,
        'num_self_play': 100,
        'train_steps_per_iter': 20000,
        'eval_win_margin': 0.05,
        'num_eval_matches': 30,
        'num_epochs': 30,
        "num_workers": os.cpu_count()
    }

    train_args.update(mcts_args)

    board_size = 8
    # Extra action for "pass"
    action_size = board_size * board_size + 1

    # policy = FastOthelloNet(board_size, action_size)
    policy = torch.load("othello_policy_RL_best.pt")
    alphaZero = Trainer(board_size, train_args, policy)

    alphaZero.train()
