import time
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import numpy as np
from multiprocessing import get_context
from typing import Deque, List, Tuple
from collections import deque
import random

from eval import evaluate_models_parallel, _worker_init
from self_play_worker import one_self_play
from Models import FastOthelloNet, AlphaZeroNet
from envs.othello import get_random_symmetry

# TODO: Clean the code, add typing, add dosctring

# TODO: why can't we beat the rollout 100% consistently?
# model is too weak? target is too noisy?
# probably since we still cannot beat the supervised model
# we need to adapt the noiseness of the gt

# TODO: add eval for 2 actors: benchmark and rollout,
# bencmark comes after we start to beat the shit out ouf rollout
# TODO: increase number of simulations during the eval to 400

# TODO: λ-return bootstrapping  as a target
# let's check how noisy the target is. Compare it to supervised
# Clear right-hand tail peaking at variance ≈ 1.0
# ➤ There are many positions (dozens!) that were visited multiple times
# but received totally contradictory labels — i.e.,
# the exact same board state ended in both a win and a loss.
# These contradictory labels show your value function is being trained on maximally noisy signals.
# If your MCTS is shallow (e.g., 100 sims), it might frequently pick different suboptimal lines
# from the same state — hence same board → different outcome → variance = 1.0.
# But if the search is deeper (e.g., 800 sims), it may consistently pick the best move in
# that position → repeat play yields the same result → lower variance.
# if the variance histogram doesn't change...
# That would suggest:
# Your MCTS is already deep enough to saturate its impact
# Or that the policy prior is too weak, so MCTS can't recover
# Or that exploration noise (Dirichlet, temperature, etc.) is dominating play
# *the deeper search does not help (100->400sim)

# I noticed that the model does not converge anymore after some point: it cannot
# beat rollout policy constantly. I took this model and did a play against a supervised
# model and plotted value function through the steps of the game, once can see there that
# only after 30 moves its value function starts to correlate with the real value function
# if you increase number of MCTS simulations, then we could reduce this nubmer to 20 steps.
# This basically says that model is unreliable at the beginning of the game. But there is a room
# for improvement, be checking the target that we train value function for we can observe if
# this target is confusing or not. We check the variance and see that for the current trained model
# self-play still produces states that are contraversial with 70% of states in the training having this
# behaviour. In the same time 0 temperature and no Dirichlet noise gives ~35% states being contraversial, hence it make sense to
# try the strategy similar to AlphaZero with a few initial steps have temperature 1 to keep exploration,
# and then switch it to 0. By running the same experiment with 8 steps we get 45% unique states having
# value variance. So, it probably makes sense to try this strategy.

# With mcts_temperature = 1.0 for the entire game the root visit count is usually flat (πᵢ ≈ 1/|A|).
# The cross‑entropy target the policy head sees is therefore almost random, so the network
# learns to imitate exploration noise instead of the best move.
# That explains
# policy loss stuck ≈ 0.8 (only a little better than uniform ≈ ln |A|)
# large variance in the value head (conflicting returns for the same canonical state)

# TODO: to get the most out of the collected data
# and utilize the GPU more since it is not a bottleneck
# (training time ~12sec, data collection ~80sec) we can
# add augmentations to the training and increase number of epochs
# we also remove repeating states to focus more on the interesting states during training
# but not on the initial non-relevant moves.

# TODO: elaborate on the policy collapse, if the exploration is too low during self-play,
# then the policy learns to beat the current policy only on the narrow set of states
# then during the evaluation we again test it against greedy policy and we know
# how to beat it, so we promote the policy, but then this policy would lose to the 
# policy few iterations ago.

# TODO: try as a learning target to use q + z, instead of z
# this should help to reduce the noise, also if we have duplicating data after data collection
# maybe it makes sense to filter it? Also maybe it makes sense to construct td-lambda target?

# TODO: add to the blog explanation on why we choose 100 simulations in the tree
# it feels like starting from this number we have good estimation of the win
# variance is lower

# Gating: Only accept a new network if it beats the previous one ≥ 55 %. Stops drifting into bad local minima.

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

        return (torch.from_numpy(np.ascontiguousarray(s)),
                torch.from_numpy(np.ascontiguousarray(p)),
                torch.as_tensor(v, dtype=torch.float32))


class Trainer:

    def __init__(self, board_size, args, policy, benchmark_policy=None):
        self.board_size = board_size
        self.args = args
        self.policy = policy
        self.benchmark_policy = benchmark_policy
        self.curr_benchmark_policy = None
        self.num_simulations = self.args['num_simulations']

        self.train_device = torch.device("cuda")
        # important to copy, otherwise we update both models during training
        self.best_policy = copy.deepcopy(self.policy).eval()

        self.value_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.args['lr'],
            weight_decay=self.args['weight_decay'])

        self.current_training_data = []
        self.dataloader = None
        buf_size = self.args.get("replay_buffer_size", 100000)
        self.replay_buffer: Deque[Tuple[np.ndarray, np.ndarray,
                                        float]] = deque(maxlen=buf_size)

        self.metrics_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'current_win_rate': [],
            'best_win_rate': [],
            'benchmark_win_rate': [],
            'current_best_win_rate': []
        }

        ctx = get_context("forkserver")
        self.manager = ctx.Manager()
        time.sleep(0.5)
        # persistent cache
        self.shared_cache = self.manager.dict()

    def clean_training_data(self):
        self.current_training_data = []

    def _extend_buffer(
            self, new_trajs: List[Tuple[np.ndarray, np.ndarray,
                                        float]]) -> None:
        """Append new (state, π, z) tuples into the replay buffer."""
        self.replay_buffer.extend(new_trajs)

    def setup_dataloader(self):
        """
        Convert self.current_training_data (list of (state, policy, value))
        into a PyTorch DataLoader for training.
        """

        states = []
        policies = []
        values = []
        for (s, p, v) in self.replay_buffer:
            states.append(s)  # 2D array: shape (board_size, board_size)
            policies.append(p)  # policy vector, shape (action_size,)
            values.append(v)  # scalar

        print(f"Collected {len(states)} unique positions")
        states = np.array(states, dtype=np.float32)
        policies = np.array(policies, dtype=np.float32)
        values = np.array(values, dtype=np.float32).reshape(-1, 1)
        dataset = RandomSymmetryDataset(states, policies, values)
        self.dataloader = DataLoader(dataset,
                                     batch_size=self.args['batch_size'],
                                     shuffle=True)

    def collect_self_play_games(self):
        """Run `num_self_play` games in parallel and extend
        `self.current_training_data`."""
        ctx = get_context("forkserver")
        state = self.best_policy.state_dict()
        base_seed = np.random.randint(1_000_000)
        # there is no need to clean this cache until we
        # find a new best model that does self-play
        with ctx.Pool(self.args["num_workers"],
                      initializer=_worker_init,
                      initargs=(base_seed, ),
                      maxtasksperchild=100) as pool:
            # Prepare a tuple of arg‑tuples so we can stream them
            work_items = [(self.board_size, self.args,
                           self.best_policy.state_dict(), self.shared_cache)
                          for _ in range(self.args["num_self_play"])]

            # chunksize=1 → each worker returns as soon as it finishes
            for traj in pool.imap_unordered(one_self_play,
                                            work_items,
                                            chunksize=1):
                self._extend_buffer(traj)

    @torch.no_grad()
    def _probe_value_bias(self, sample_size: int = 1024):
        """
        1. Randomly pick `sample_size` indices from the *same* Dataset
        (`self.dataloader.dataset`).
        2. Call dataset[idx] so every item goes through the identical
        random‑symmetry augmentation.
        3. Return mean bias and MSE of the value head.
        """
        ds = self.dataloader.dataset
        idxs = random.sample(range(len(ds)), k=min(sample_size, len(ds)))

        states, targets_v = [], []
        for i in idxs:
            s, _, v = ds[i]  # ds[i] already returns tensors
            states.append(s)
            targets_v.append(v)

        states = torch.stack(states, 0).to(self.train_device)
        targets_v = torch.stack(targets_v,
                                0).unsqueeze(1).to(self.train_device)

        _, pred_v = self.policy(states)
        diff = pred_v - targets_v
        return diff.mean().item(), (diff**2).mean().item()
    
    @torch.no_grad()
    def _probe_entropy(self, sample_size: int = 1024):
        """
        Random mini‑batch → net‑policy entropy  +  MCTS‑target entropy.
        """
        ds = self.dataloader.dataset
        idxs = random.sample(range(len(ds)), k=min(sample_size, len(ds)))

        states, target_pi = [], []
        for i in idxs:
            s, π̂, _ = ds[i]
            states.append(s)
            target_pi.append(π̂)

        states    = torch.stack(states, 0).to(self.train_device)
        target_pi = torch.stack(target_pi, 0).to(self.train_device)

        logits, _ = self.policy(states)
        p_net     = torch.softmax(logits, dim=-1)

        H_net = -(p_net     * torch.log(p_net + 1e-12)).sum(dim=-1).mean().item()
        H_tgt = -(target_pi * torch.log(target_pi + 1e-12)).sum(dim=-1).mean().item()
        return H_net, H_tgt

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
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
            self.optimizer.step()

            total_policy_loss += policy_loss.cpu().item()
            total_value_loss += value_loss.cpu().item()
            total_batches += 1

        avg_policy_loss = total_policy_loss / total_batches
        avg_value_loss = total_value_loss / total_batches
        avg_total_loss = avg_policy_loss + avg_value_loss

        val_bias, val_mse     = self._probe_value_bias()
        H_net, H_target       = self._probe_entropy()

        self.metrics_history['policy_loss'].append(avg_policy_loss)
        self.metrics_history['value_loss'].append(avg_value_loss)
        self.metrics_history['total_loss'].append(avg_total_loss)

        self.metrics_history.setdefault('value_bias',  []).append(val_bias)
        self.metrics_history.setdefault('value_mse',   []).append(val_mse)
        self.metrics_history.setdefault('net_entropy', []).append(H_net)
        self.metrics_history.setdefault('mcts_entropy',[]).append(H_target)

        print(
            f"Policy L={avg_policy_loss:.4f}  Value L={avg_value_loss:.4f}  "
            f"Δv={val_bias:+.3f}  MSE={val_mse:.3f}  "
            f"H_net={H_net:.2f}  H_mcts={H_target:.2f}"
        )

    def train(self):
        for _ in range(self.args['num_iterations']):

            # self-play
            start_time = time.time()
            self.collect_self_play_games()

            print("Collection Time taken", time.time() - start_time)

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
            self.shared_cache.clear()
            torch.save(self.best_policy, 'othello_policy_RL.pt')

            current_win_rate, best_win_rate = evaluate_models_parallel(
                self.board_size,
                self.args,
                self.policy.state_dict(),
                self.curr_benchmark_policy.state_dict() if self.curr_benchmark_policy is not None else None,
                n_matches=30)
            self.metrics_history['benchmark_win_rate'].append(current_win_rate)
            self.metrics_history['current_best_win_rate'].append(best_win_rate)
            print(
                f"Eval against bencmark: Win Rate Current {current_win_rate*100:.1f}% vs Win rate Benchmark: {best_win_rate*100:.1f}%"
            )

            if self.curr_benchmark_policy is None and current_win_rate > 0.9:
                print("🎉  Rollout beaten (>90 %). "
                        "Promoting given benchmark net as benchmark and "
                        "increasing simulations to 400.")
                self.curr_benchmark_policy = self.benchmark_policy
                self.num_simulations = 400
                self.args['num_simulations'] = 400

                new_lr = 3e-4
                for pg in self.optimizer.param_groups:   # <‑‑ add these
                    pg['lr'] = new_lr                    #     three lines
                self.args['lr'] = new_lr



if __name__ == '__main__':
    mcts_args = {
        'c_puct': 2.0,
        'num_simulations': 100,
        'dirichlet_alpha': 1.0,
        'dirichlet_epsilon': 0.3
    }

    train_args = {
        'lr': 1e-3,
        'weight_decay': 3e-6,
        'batch_size': 256,
        'mcts_temperature': 1.0,
        'num_exploratory_moves': 35,
        'num_iterations': 2000,
        'num_self_play': 300,
        'replay_buffer_size': int(2e5),
        'train_steps_per_iter': 5000,
        'eval_win_margin': 0.1,
        'num_eval_matches': 50,
        'num_epochs': 60,
        "num_workers": os.cpu_count()
    }

    train_args.update(mcts_args)

    board_size = 8
    # Extra action for "pass"
    action_size = board_size * board_size + 1

    # policy = FastOthelloNet(board_size, action_size)
    # TODO: add both rollout and bencmark policy to the eval
    # policy = torch.load("othello_policy_RL.pt")
    bencnmark_policy = AlphaZeroNet(board_size, action_size, 10, 128)  # create model instance
    bencnmark_policy.load_state_dict(torch.load('othello_policy_supervised_v7_best.pt'))
    bencnmark_policy.eval()
    
    policy = AlphaZeroNet(board_size, action_size, 10, 128)
    alphaZero = Trainer(board_size, train_args, policy, bencnmark_policy)
    alphaZero.train()

    # current_win_rate, best_win_rate = evaluate_models_parallel(
    #     board_size,
    #     train_args,
    #     bencnmark_policy.state_dict(),
    #     None,
    #     n_matches=150)
    # print(
    #     f"Eval against bencmark: Win Rate Current {current_win_rate*100:.1f}% vs Win rate Benchmark: {best_win_rate*100:.1f}%"
    # )

    # alphaZero.collect_self_play_games()
    # raw = alphaZero.current_training_data
    # print(
    #     f"Collected {len(raw):,} positions from {train_args['num_self_play']} games"
    # )

    # import pickle, pathlib, hashlib
    # import pandas as pd

    # suffix = f"{train_args['num_self_play']}self-play_{train_args['num_simulations']}sim_lambda9"
    # out_dir = pathlib.Path("diagnostics")
    # out_dir.mkdir(exist_ok=True)
    # with open(out_dir / f"training_set_{suffix}.pkl", "wb") as f:
    #     pickle.dump(raw, f)

    # def hash_state(board: np.ndarray) -> str:
    #     """Fast, order-preserving hash for an 8×8 int8 board."""
    #     return hashlib.sha1(board.tobytes()).hexdigest()

    # buckets = {}
    # for s, _π, v in raw:
    #     h = hash_state(s.astype(np.int8))
    #     buckets.setdefault(h, []).append(float(v))

    # stats = []
    # for h, vals in buckets.items():
    #     vals = np.array(vals, dtype=np.float32)
    #     stats.append({
    #         "hash": h,
    #         "count": len(vals),
    #         "mean_v": vals.mean(),
    #         "var_v": vals.var(ddof=0)
    #     })
    # df = pd.DataFrame(stats)
    # df.to_csv(out_dir / f"value_stats_{suffix}.csv", index=False)
    # df = pd.read_csv(out_dir / f"value_stats_{suffix}.csv")
    # df = df[df["count"] > 1].copy()

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.hist(df["var_v"], bins=50, log=True)
    # plt.xlabel("Variance of value target per unique state")
    # plt.ylabel("Frequency (log scale)")
    # plt.title(
    #     f"Noise level of value targets after {train_args['num_self_play']} self-play games"
    # )
    # plt.tight_layout()
    # plt.savefig(out_dir / f"value_variance_hist_{suffix}.png", dpi=150)

    # noisiest = df.sort_values(by=["var_v", "count"],
    #                           ascending=[False, False]).head(10)
    # print("\nTop-10 noisiest positions:")
    # print(noisiest[["count", "mean_v", "var_v"]])

    # high_var_pct = (df["var_v"] > 0.0).mean() * 100
    # print(
    #     f"Fraction of positions with value variance > 0.0: {high_var_pct:.2f}%"
    # )
