import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import pickle, pathlib, hashlib
import pandas as pd

from MCTS_model import MCTS
from envs.othello import OthelloGameNew as OthelloGame
from Models import FastOthelloNet, AlphaZeroNet
from train import Trainer


def play_match_record_values(
    env,
    mcts_A,  # plays as +1 in this game
    mcts_B  # plays as -1
) -> Tuple[List[float], List[float], str]:

    state = env.get_initial_state()
    player = 1  # +1 always starts
    vals_A: List[float] = []
    vals_B: List[float] = []

    while True:
        valid = env.get_valid_moves(state, player)
        if valid.sum() == 0:  # no legal moves: draw
            return vals_A, vals_B, "Draw"

        if player == 1:
            action_probs = mcts_A.policy_improve_step(init_state=state,
                                                      init_player=player,
                                                      temp=0.0)
            vals_A.append((mcts_A.root.value + 1) / 2)  # −1…+1 → 0…1
        else:
            action_probs = mcts_B.policy_improve_step(init_state=state,
                                                      init_player=player,
                                                      temp=0.0)
            vals_B.append((mcts_B.root.value + 1) / 2)

        # Choose & play the move
        action = np.argmax(action_probs)
        mcts_A.make_move(action)
        mcts_B.make_move(action)
        state = env.get_next_state(state, action, player)

        reward, done = env.get_value_and_terminated(state, action, player)
        if done:
            if reward == 1:  # the *current* player won
                return vals_A, vals_B, ("A" if player == 1 else "B")
            elif reward == -1:  # current player lost ➞ opponent wins
                return vals_A, vals_B, ("B" if player == 1 else "A")
            else:
                return vals_A, vals_B, "Draw"

        player = env.get_opponent(player)


def collect_value_trajectories(env,
                               args_A,
                               policy_A,
                               args_B,
                               policy_B,
                               n_matches: int = 20):
    curves_A, curves_B, results_A, results_B = [], [], [], []
    for i in range(n_matches):
        # Alternate colours each game
        if i % 2 == 0:
            mcts_A = MCTS(env, args_A, policy_A)  # plays as +1
            mcts_B = MCTS(env, args_B, policy_B)  # plays as -1
            vals_A, vals_B, res = play_match_record_values(env, mcts_A, mcts_B)
            win_A = (res == "A")
            win_B = (res == "B")
        else:
            mcts_A = MCTS(env, args_A, policy_A)  # will play as -1
            mcts_B = MCTS(env, args_B, policy_B)  # will play as +1
            # Swap roles inside the helper:
            vals_B, vals_A, res = play_match_record_values(env, mcts_B, mcts_A)
            win_A = (res == "B")  # because mcts_A was “second” this game
            win_B = (res == "A")

        curves_A.append(vals_A)
        curves_B.append(vals_B)
        results_A.append("win" if win_A else "loss" if win_B else "draw")
        results_B.append("win" if win_B else "loss" if win_A else "draw")

    return curves_A, results_A, curves_B, results_B


def _plot_one_side(curves: List[List[float]], results: List[str], title: str):
    max_len = max(len(v) for v in curves)

    def pad(arr: np.ndarray) -> np.ndarray:
        """Right‑pad with NaN so every row has length max_len."""
        return np.concatenate([arr, np.full(max_len - len(arr), np.nan)])

    # 2.  Start a fresh figure *for this player*
    plt.figure()  # <‑‑ ONE chart per player (no subplots)

    # 3.  Plot every game in a very transparent line
    colour_map = dict(win="green", loss="red", draw="gray")
    for vals, res in zip(curves, results):
        xs = np.arange(1, len(vals) + 1)
        plt.plot(
            xs,
            vals,
            color=colour_map[res],
            alpha=0.5,  # transparency so many lines can overlap
            linewidth=1)

    # 5.  Cosmetic touches
    plt.ylim(0, 1)
    plt.xlim(1, max_len)
    plt.xlabel("move number (t)")
    plt.ylabel("P(win)")
    plt.title(title)
    plt.grid(True, alpha=0.3)


def plot_value_trajectories(curves_A, results_A, curves_B, results_B):
    _plot_one_side(curves_A, results_A, "Player A value trajectories")
    _plot_one_side(curves_B, results_B, "Player B value trajectories")


def check_random_rollout_trajectories(num_simulations=400):
    env = OthelloGame(8)
    args_A = {'c_puct': 2.0, 'num_simulations': num_simulations}
    args_B = {'c_puct': 2.0, 'num_simulations': num_simulations}

    curves_A, res_A, curves_B, res_B = collect_value_trajectories(env,
                                                                  args_A,
                                                                  None,
                                                                  args_B,
                                                                  None,
                                                                  n_matches=5)

    plot_value_trajectories(curves_A, res_A, curves_B, res_B)
    plt.show()


def collect_and_analyse_training_data():
    board_size = 8
    policy = FastOthelloNet(board_size, board_size * board_size + 1)
    policy.load_state_dict(torch.load('othello_policy_RL_fast_current.pt'))

    train_args = {
        'c_puct': 2.0,
        'num_simulations': 100,
        'dirichlet_alpha': 1.0,
        'dirichlet_epsilon': 0.3,
        'mcts_temperature': 1.0,
        'num_exploratory_moves': 35,
        'num_self_play': 100,
        'num_workers': os.cpu_count(),
        'lambda': 1.0,
        'replay_buffer_path': 'replay_buffer_test.pkl'
    }

    alphaZero = Trainer(board_size, train_args, policy)

    alphaZero.collect_self_play_games()
    raw = alphaZero.replay_buffer

    suffix = f"{train_args['num_self_play']}self-play_{train_args['num_simulations']}sim_lambda_{train_args['lambda']}"
    out_dir = pathlib.Path("diagnostics")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"training_set_{suffix}.pkl", "wb") as f:
        pickle.dump(raw, f)

    def hash_state(board: np.ndarray) -> str:
        """Fast, order-preserving hash for an 8×8 int8 board."""
        return hashlib.sha1(board.tobytes()).hexdigest()

    buckets = {}
    for s, _, v, _ in raw:
        h = hash_state(s.astype(np.int8))
        buckets.setdefault(h, []).append(float(v))

    stats = []
    for h, vals in buckets.items():
        vals = np.array(vals, dtype=np.float32)
        stats.append({
            "hash": h,
            "count": len(vals),
            "mean_v": vals.mean(),
            "var_v": vals.var(ddof=0)
        })
    df = pd.DataFrame(stats)
    df.to_csv(out_dir / f"value_stats_{suffix}.csv", index=False)
    df = pd.read_csv(out_dir / f"value_stats_{suffix}.csv")
    df = df[df["count"] > 1].copy()

    plt.figure()
    plt.hist(df["var_v"], bins=50, log=True)
    plt.xlabel("Variance of value target per unique state")
    plt.ylabel("Frequency (log scale)")
    plt.title(
        f"Noise level of value targets after {train_args['num_self_play']} self-play games"
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"value_variance_hist_{suffix}.png", dpi=150)

    noisiest = df.sort_values(by=["var_v", "count"],
                              ascending=[False, False]).head(10)
    print("\nTop-10 noisiest positions:")
    print(noisiest[["count", "mean_v", "var_v"]])

    high_var_pct = (df["var_v"] > 0.0).mean() * 100
    print(
        f"Fraction of positions with value variance > 0.0: {high_var_pct:.2f}%"
    )


if __name__ == "__main__":
    collect_and_analyse_training_data()
