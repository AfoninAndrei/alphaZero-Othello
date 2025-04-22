import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

import torch
import numpy as np

from MCTS_model import MCTS
from envs.othello import OthelloGame

import time


@torch.no_grad()
def evaluate_models(env, args, policy, best_policy, n_matches=20):
    """
    Compare mcts_a vs mcts_b by playing n_matches in the given environment.
    We alternate colors (i.e. which tree plays as +1) by swapping the trees every game.
    Returns a tuple:
        (win_rate for mcts_a, win_rate for mcts_b)
    """
    wins_a = 0
    wins_b = 0
    for i in range(n_matches):

        if i % 2 == 0:
            # Even game: mcts_a plays as +1, mcts_b as -1.
            result = play_match(env, MCTS(env, args, policy),
                                MCTS(env, args, best_policy))
            if result == "A":
                wins_a += 1
            elif result == "B":
                wins_b += 1
        else:
            # Odd game: swap roles: mcts_b plays as +1, mcts_a as -1.
            result = play_match(env, MCTS(env, args, best_policy),
                                MCTS(env, args, policy))
            # In this case, "A" means the first tree (mcts_b) wins,
            # so from mcts_a’s perspective, that is a loss.
            if result == "A":
                wins_b += 1
            elif result == "B":
                wins_a += 1
    return wins_a / n_matches, wins_b / n_matches


def play_match(env, mcts_first, mcts_second):
    """
    Play a single game in the given environment with:
      - mcts_first playing as +1
      - mcts_second playing as -1.

    Returns:
      "A" if the player using mcts_first wins,
      "B" if the player using mcts_second wins,
      "Draw" otherwise.
    """
    state = env.get_initial_state()
    player = 1  # always start with +1

    while True:
        valid_moves = env.get_valid_moves(state, player)
        if valid_moves.sum() == 0:
            return "Draw"  # no legal moves means a draw

        if player == 1:
            action_probs = mcts_first.policy_improve_step(init_state=state,
                                                          init_player=player,
                                                          temp=0.0)
        else:
            action_probs = mcts_second.policy_improve_step(init_state=state,
                                                           init_player=player,
                                                           temp=0.0)
        action = np.argmax(action_probs)
        # update the root node of the trees
        mcts_first.make_move(action)
        mcts_second.make_move(action)
        state = env.get_next_state(state, action, player)

        reward, done = env.get_value_and_terminated(state, action, player)
        if done:
            if reward == 1:
                # current player wins
                return "A" if player == 1 else "B"
            elif reward == -1:
                # current player loses; so opponent wins
                return "B" if player == 1 else "A"
            else:
                return "Draw"
        player = env.get_opponent(player)


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
            alpha=0.15,  # transparency so many lines can overlap
            linewidth=1)

    # 4.  Stack padded arrays and compute per‑move means for wins & losses
    wins = [pad(np.asarray(v)) for v, r in zip(curves, results) if r == "win"]
    losses = [
        pad(np.asarray(v)) for v, r in zip(curves, results) if r == "loss"
    ]

    xs_full = np.arange(1, max_len + 1)
    if wins:  # green, bold, opaque
        mean_win = np.nanmean(wins, axis=0)
        plt.plot(xs_full,
                 mean_win,
                 color="green",
                 linewidth=2.5,
                 label="mean (wins)")
    if losses:  # red, bold, opaque
        mean_loss = np.nanmean(losses, axis=0)
        plt.plot(xs_full,
                 mean_loss,
                 color="red",
                 linewidth=2.5,
                 label="mean (losses)")

    # 5.  Cosmetic touches
    plt.ylim(0, 1)
    plt.xlim(1, max_len)
    plt.xlabel("move number (t)")
    plt.ylabel("P(win)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")


def plot_value_trajectories(curves_A, results_A, curves_B, results_B):
    _plot_one_side(curves_A, results_A, "Player A value trajectories")
    _plot_one_side(curves_B, results_B, "Player B value trajectories")


if __name__ == "__main__":
    args = {'c_puct': 2.0, 'num_simulations': 200, 'mcts_temperature': 1.0}
    board_size = 8
    env = OthelloGame(board_size)
    model_path = "othello_policy_RL.pt"
    policy = torch.load(model_path)

    model_path_supervised = "othello_policy_supervised.pt"
    policy_supervised = torch.load(model_path_supervised)
    mcts = MCTS(env, args, policy)

    args_opponent = {
        'c_puct': 1.0,
        'num_simulations': 100,
        'mcts_temperature': 1.0
    }

    mcts_opponent = MCTS(env, args_opponent, policy_supervised)

    start_time = time.time()
    print(play_match(env, mcts, mcts_opponent))
    print('Time taken', time.time() - start_time)

    # TODO: Compare the curves for the model vs model with MCTS - this should show
    # how far it is from being optimal?
    # let's compare it to the MCTS prediction rollout and supervised model

    # args_A = {'c_puct': 2.0, 'num_simulations': 2, 'mcts_temperature': 1.0}
    # args_B = {'c_puct': 2.0, 'num_simulations': 2, 'mcts_temperature': 1.0}

    # curves_A, res_A, curves_B, res_B = collect_value_trajectories(
    #     env, args_A, policy, args_B, policy_supervised, n_matches=5)

    # plot_value_trajectories(curves_A, res_A, curves_B, res_B)
    # plt.show()
