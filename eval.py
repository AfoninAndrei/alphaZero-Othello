import os
import random
import numpy as np
from multiprocessing import get_context
import torch

from MCTS_model import MCTS
from envs.othello import OthelloGameNew as OthelloGame
from Models import FastOthelloNet, AlphaZeroNet


@torch.no_grad()
def evaluate_models(board_size, args, policy, best_policy, n_matches=20):
    """
    Compare mcts_a vs mcts_b by playing n_matches in the given environment.
    We alternate colors (i.e. which tree plays as +1) by swapping the trees every game.
    Returns a tuple:
        (win_rate for mcts_a, win_rate for mcts_b)
    """
    wins_a = 0
    wins_b = 0
    env = OthelloGame(board_size)
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


@torch.no_grad()
def evaluate_models_parallel(board_size,
                             args,
                             policy_state,
                             best_policy_state,
                             n_matches=20):
    ctx = get_context("forkserver")
    wins_a = 0
    wins_b = 0
    base_seed = np.random.randint(1_000_000)  # any seed you like

    with ctx.Pool(args["num_workers"],
                  initializer=_worker_init,
                  initargs=(base_seed, ),
                  maxtasksperchild=100) as pool:
        # Prepare a tuple of arg‑tuples so we can stream them
        work_items = [(match_idx, board_size, args, policy_state,
                       best_policy_state) for match_idx in range(n_matches)]

        # chunksize=1 → each worker returns as soon as it finishes
        for winner in pool.imap_unordered(_run_one_match,
                                          work_items,
                                          chunksize=1):
            if winner == "A":
                wins_a += 1
            elif winner == "B":
                wins_b += 1

    return wins_a / n_matches, wins_b / n_matches


def _worker_init(seed_base):
    # one-time initialisation – this runs once per process
    # we do not want every game starts from an identical RNG state;
    # on small boards this can reduce diversity badly for both training/eval
    pid = os.getpid()
    np.random.seed(seed_base + pid)
    random.seed(seed_base + pid)
    torch.manual_seed(seed_base + pid)


def _run_one_match(args_tuple):
    """
    Play one match with index i, returns ("A" or "B").
    We embed the even/odd logic here.
    env_ctor: zero‐arg callable that returns a fresh env
    """
    worker_id, board_size, args, policy_state, best_policy_state = args_tuple

    # ----- reconstruct lightweight objects inside the worker
    env = OthelloGame(board_size)

    policy, best_policy = None, None

    if policy_state is not None:
        (policy_class, policy_config, policy_state_dict) = policy_state
        policy = policy_class(**policy_config)
        policy.load_state_dict(policy_state_dict)
        policy.eval()

    if best_policy_state is not None:
        (best_policy_class, best_policy_config,
         best_policy_state_dict) = best_policy_state
        best_policy = best_policy_class(**best_policy_config)
        best_policy.load_state_dict(best_policy_state_dict)
        best_policy.eval()

    # choose which policy is first based on i
    if worker_id % 2 == 0:
        mcts_first = MCTS(env, args, policy)
        mcts_second = MCTS(env, args, best_policy)
        invert = False
    else:
        mcts_first = MCTS(env, args, best_policy)
        mcts_second = MCTS(env, args, policy)
        invert = True

    # play the match
    result = play_match(env, mcts_first, mcts_second)

    if invert:
        if result == "A":
            return "B"
        elif result == "B":
            return "A"
    return result


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

        mcts_first.make_move(action)
        mcts_second.make_move(action)
        player = env.get_opponent(player)


if __name__ == "__main__":
    board_size = 8
    env = OthelloGame(board_size)

    # model_path_rl = "othello_policy_RL_big.pt"
    # policy = AlphaZeroNet(board_size, 65, 5, 128)
    model_path_rl = "othello_policy_RL_small.pt"
    policy = FastOthelloNet(board_size, 65)
    policy.load_state_dict(torch.load(model_path_rl))
    policy.eval()
    args = {'c_puct': 3.0, 'num_simulations': 800}
    mcts = MCTS(env, args, policy, apply_symmetry=True)

    # model_path_supervised = "othello_policy_supervised_big.pt"
    # policy_supervised = AlphaZeroNet(board_size, 65, 5, 128)
    model_path_supervised = "othello_policy_supervised_small.pt"
    policy_supervised = FastOthelloNet(board_size, 65)
    policy_supervised.load_state_dict(torch.load(model_path_supervised))
    policy_supervised.eval()
    args_opponent = {'c_puct': 3.0, 'num_simulations': 800}
    mcts_opponent = MCTS(env,
                         args_opponent,
                         policy_supervised,
                         apply_symmetry=True)

    print("Winner is ", play_match(env, mcts, mcts_opponent))

    mcts = MCTS(env, args, policy, apply_symmetry=True)
    mcts_opponent = MCTS(env,
                         args_opponent,
                         policy_supervised,
                         apply_symmetry=True)

    print("Winner is ", play_match(env, mcts_opponent, mcts))
