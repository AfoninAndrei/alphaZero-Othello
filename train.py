import time
import os
import pickle

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
import hashlib

from eval import evaluate_models_parallel, _worker_init
from self_play_worker import one_self_play
from Models import FastOthelloNet, AlphaZeroNet
from envs.othello import get_random_symmetry


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

    def __init__(self,
                 board_size,
                 args,
                 policy,
                 self_play_policy=None,
                 benchmark_policy=None):
        self.board_size = board_size
        self.args = args
        self.policy = policy
        self.benchmark_policy = benchmark_policy
        self.curr_benchmark_policy = None
        self.num_simulations = self.args['num_simulations']

        self.train_device = torch.device(
            "mps" if torch.backends.mps.is_available(
            ) else "cuda" if torch.cuda.is_available() else "cpu")
        # important to copy, otherwise we update both models during training
        if self_play_policy is None:
            self.best_policy = copy.deepcopy(self.policy).eval()
        else:
            self.best_policy = self_play_policy
        self.model_version = 0

        self.value_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self.args.get('lr', 1e-3),
                                          weight_decay=self.args.get(
                                              'weight_decay', 3e-6))

        self.dataloader = None
        buf_size = self.args.get("replay_buffer_size", 500000)
        self.buffer_path = self.args.get('replay_buffer_path', None)
        self.replay_buffer: Deque[Tuple[np.ndarray, np.ndarray, float,
                                        int]] = deque(maxlen=buf_size)
        # this should update the model version
        self._load_replay_buffer()

        self.metrics_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'current_win_rate': [],
            'best_win_rate': [],
            'benchmark_win_rate': [],
            'current_best_win_rate': []
        }

        ctx = get_context("spawn")
        self.manager = ctx.Manager()
        time.sleep(0.5)
        # persistent cache
        self.shared_cache = self.manager.dict()

    def _hash_state(self, board: np.ndarray) -> str:
        """Fast, order-preserving hash for an int8 board."""
        return hashlib.sha1(board.astype(np.int8).tobytes()).hexdigest()

    def _save_replay_buffer(self) -> None:
        """Pickle the *entire* replay_buffer to disk (atomic write)."""
        tmp_path = self.buffer_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(self.replay_buffer,
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, self.buffer_path)

    def _load_replay_buffer(self) -> None:
        """Populate self.replay_buffer from an existing pickle file."""
        if not self.buffer_path:
            return
        try:
            with open(self.buffer_path, "rb") as f:
                loaded = pickle.load(f)
            # Keep user-defined maxlen
            self.replay_buffer.clear()
            self.replay_buffer.extend(loaded)
            print(f"Loaded {len(self.replay_buffer):,} samples from "
                  f"'{self.buffer_path}'")

            versions = [
                traj[3] for traj in self.replay_buffer
                if isinstance(traj, (tuple, list)) and len(traj) >= 4
            ]
            self.model_version = max(versions, default=0)
            print(f"Latest model version is {self.model_version}")
        except Exception as err:
            print(
                f"⚠️  Could not load replay buffer ({err!s}); starting fresh.")

    def _extend_buffer(
            self, new_trajs: List[Tuple[np.ndarray, np.ndarray,
                                        float]]) -> None:
        ver = self.model_version
        self.replay_buffer.extend((s, p, v, ver) for (s, p, v) in new_trajs)

    def _aggregate_duplicates(
            self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Collapse duplicates that stem from the *same* model version.
        Returns three aligned Python lists (states, avg-pi, avg-v).
        """
        buckets = {}  # key = (hash, version)
        for (s, pi, v, ver) in self.replay_buffer:
            key = (self._hash_state(s), ver)
            if key not in buckets:
                buckets[key] = {
                    "state": s,
                    "sum_pi": pi.copy(),
                    "sum_v": v,
                    "count": 1,
                }
            else:
                b = buckets[key]
                b["sum_pi"] += pi
                b["sum_v"] += v
                b["count"] += 1

        states, policies, values = [], [], []
        for b in buckets.values():
            cnt = b["count"]
            avg_pi = b["sum_pi"] / cnt  # arithmetic mean of dists
            avg_pi /= avg_pi.sum() + 1e-12  # re-normalise, safe
            v = b["sum_v"] / cnt
            states.append(b["state"])
            policies.append(avg_pi.astype(np.float32))
            values.append(np.float32(v))
        return states, policies, values

    def setup_dataloader(self):
        """
        Convert self.current_training_data (list of (state, policy, value))
        into a PyTorch DataLoader for training.
        """
        states, policies, values = self._aggregate_duplicates()
        print(f"Collected {len(states)} unique (state,version) pairs "
              f"from {len(self.replay_buffer):,} raw samples")

        states = np.array(states, dtype=np.float32)
        policies = np.array(policies, dtype=np.float32)
        values = np.array(values, dtype=np.float32).reshape(-1, 1)
        dataset = RandomSymmetryDataset(states, policies, values)
        self.dataloader = DataLoader(dataset,
                                     batch_size=self.args['batch_size'],
                                     num_workers=self.args['num_workers'],
                                     shuffle=True)

        self.probe_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args['batch_size'],
            shuffle=True,
            num_workers=0)

    def collect_self_play_games(self):
        """Run `num_self_play` games in parallel and extend
        `self.current_training_data`."""
        ctx = get_context("spawn")
        time.sleep(1.0)
        base_seed = np.random.randint(1_000_000)
        # there is no need to clean this cache until we
        # find a new best model that does self-play
        best_policy = (self.best_policy.__class__,
                       self.best_policy.get_config(),
                       self.best_policy.state_dict())
        with ctx.Pool(self.args["num_workers"],
                      initializer=_worker_init,
                      initargs=(base_seed, ),
                      maxtasksperchild=100) as pool:
            # Prepare a tuple of arg‑tuples so we can stream them
            work_items = [(self.board_size, self.args, best_policy,
                           self.shared_cache)
                          for _ in range(self.args["num_self_play"])]

            # chunksize=1 → each worker returns as soon as it finishes
            for traj in pool.imap_unordered(one_self_play,
                                            work_items,
                                            chunksize=1):
                self._extend_buffer(traj)

        self._save_replay_buffer()

    @torch.no_grad()
    def _probe_value_bias(self):
        """
        1. Randomly pick `sample_size` indices from the *same* Dataset
        (`self.dataloader.dataset`).
        2. Call dataset[idx] so every item goes through the identical
        random‑symmetry augmentation.
        3. Return mean bias and MSE of the value head.
        """
        states, _, targets_v = next(iter(self.probe_loader))
        states = states.to(self.train_device)
        targets_v = targets_v.to(self.train_device)

        _, pred_v = self.policy(states)
        diff = pred_v - targets_v
        return diff.mean().item()

    @torch.no_grad()
    def _probe_entropy(self):
        """
        Random mini‑batch → net‑policy entropy  +  MCTS‑target entropy.
        """
        states, target_pi, _ = next(iter(self.probe_loader))
        states = states.to(self.train_device)
        target_pi = target_pi.to(self.train_device)

        logits, _ = self.policy(states)
        p_net = torch.softmax(logits, dim=-1)

        H_net = -(p_net * torch.log(p_net + 1e-12)).sum(dim=-1).mean().item()
        H_tgt = -(target_pi *
                  torch.log(target_pi + 1e-12)).sum(dim=-1).mean().item()
        return H_net, H_tgt

    def train_iters(self):
        self.policy.to(self.train_device)
        self.policy.train()
        total_policy_loss = total_value_loss = total_entropy_loss = 0.0
        total_batches = 0

        lambda_entropy = self.args["entropy_coef"]

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
            probs = torch.softmax(policy_logits, dim=-1)
            entropy = torch.mean(torch.sum(-probs * log_probs, dim=-1))  # H(π)
            entropy_loss = -lambda_entropy * entropy  # <‑‑ new
            # compute value loss
            value_loss = self.value_loss(value, target_value)

            loss = policy_loss + value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
            self.optimizer.step()

            total_policy_loss += policy_loss.cpu().item()
            total_value_loss += value_loss.cpu().item()
            total_entropy_loss += -entropy_loss.detach().cpu().item()
            total_batches += 1

        avg_policy_loss = total_policy_loss / total_batches
        avg_value_loss = total_value_loss / total_batches
        avg_entropy_loss = total_entropy_loss / total_batches
        avg_total_loss = avg_policy_loss + avg_value_loss

        val_bias = self._probe_value_bias()
        H_net, H_target = self._probe_entropy()

        self.metrics_history['policy_loss'].append(avg_policy_loss)
        self.metrics_history['value_loss'].append(avg_value_loss)
        self.metrics_history['total_loss'].append(avg_total_loss)

        self.metrics_history.setdefault('avg_entropy',
                                        []).append(avg_entropy_loss)
        self.metrics_history.setdefault('value_bias', []).append(val_bias)
        self.metrics_history.setdefault('net_entropy', []).append(H_net)
        self.metrics_history.setdefault('mcts_entropy', []).append(H_target)

        print(f"Policy L={avg_policy_loss:.4f}  "
              f"Value L={avg_value_loss:.4f}  "
              f"H_bonus=×{avg_entropy_loss:.2f}  "
              f"Δv={val_bias:+.3f} "
              f"H_net={H_net:.2f}  H_mcts={H_target:.2f}")

    def train(self):
        for _ in range(self.args['num_iterations']):

            # self-play
            start_time = time.time()
            self.collect_self_play_games()

            print("Collection Time taken", time.time() - start_time)

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
            self.args, (self.policy.__class__, self.policy.get_config(),
                        self.policy.state_dict()),
            (self.best_policy.__class__, self.best_policy.get_config(),
             self.best_policy.state_dict()),
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
            self.model_version += 1
            self.shared_cache.clear()
            torch.save(self.best_policy.state_dict(),
                       'othello_policy_RL_intermediate.pt')

            current_win_rate, best_win_rate = evaluate_models_parallel(
                self.board_size,
                self.args, (self.policy.__class__, self.policy.get_config(),
                            self.policy.state_dict()),
                (self.curr_benchmark_policy.__class__,
                 self.curr_benchmark_policy.get_config(),
                 self.curr_benchmark_policy.state_dict())
                if self.curr_benchmark_policy is not None else None,
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
                self.num_simulations = 300
                self.args['num_simulations'] = 300

                new_lr = 2e-4
                for pg in self.optimizer.param_groups:
                    pg['lr'] = new_lr
                self.args['lr'] = new_lr


if __name__ == '__main__':
    mcts_args = {
        'c_puct': 2.0,
        'num_simulations': 300,
        'dirichlet_alpha': 1.0,
        'dirichlet_epsilon': 0.3
    }

    train_args = {
        'lr': 5e-4,
        'weight_decay': 3e-6,
        'batch_size': 256,
        'mcts_temperature': 1.0,
        'num_exploratory_moves': 35,
        'num_iterations': 200,
        'num_self_play': 300,
        'replay_buffer_size': int(1e6),
        'replay_buffer_path': 'replay_buffer_old.pkl',
        'train_steps_per_iter': 5000,
        'eval_win_margin': 0.1,
        'num_eval_matches': 50,
        'num_epochs': 80,
        'num_workers': os.cpu_count(),
        'entropy_coef': 0.01,
        'lambda': 0.98
    }

    train_args.update(mcts_args)

    board_size = 8
    # Extra action for "pass"
    action_size = board_size * board_size + 1

    policy = AlphaZeroNet(board_size, action_size, 5, 128)
    bencnmark_policy = AlphaZeroNet(board_size, action_size, 5,
                                    128)  # create model instance
    bencnmark_policy.load_state_dict(
        torch.load('othello_policy_supervised_big.pt'))
    bencnmark_policy.eval()

    self_play_policy = FastOthelloNet(board_size, action_size)
    self_play_policy.load_state_dict(torch.load('othello_policy_RL_small.pt'))
    self_play_policy.eval()

    alphaZero = Trainer(board_size,
                        train_args,
                        policy,
                        self_play_policy=self_play_policy,
                        benchmark_policy=bencnmark_policy)
    alphaZero.curr_benchmark_policy = alphaZero.benchmark_policy
    alphaZero.train()
