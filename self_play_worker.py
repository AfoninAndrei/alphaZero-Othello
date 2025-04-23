import random
import torch
import numpy as np

from MCTS_model import MCTS
from Models import FastOthelloNet
from envs.othello import OthelloGame


@torch.no_grad()
def one_self_play(args_tuple):
    """
    Single complete self‑play game executed in a child process.
    Returns a list of (state, improved_policy, value) triples.
    """
    worker_id, board_size, args, policy_state, seed = args_tuple
    # ----- reproducibility inside each worker
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)

    # ----- reconstruct lightweight objects inside the worker
    env = OthelloGame(board_size)
    policy = FastOthelloNet(board_size,
                            board_size * board_size + 1)  # +1 for "pass"
    policy.load_state_dict(policy_state)
    policy.eval()  # no gradients needed in self‑play

    mcts = MCTS(env,
                args,
                policy,
                dirichlet_alpha=args["dirichlet_alpha"],
                dirichlet_epsilon=args["dirichlet_epsilon"])

    trajectory = []
    state = env.get_initial_state()
    player, is_terminal = 1, False

    while not is_terminal:
        action_probs = mcts.policy_improve_step(state,
                                                player,
                                                temp=args["mcts_temperature"])
        trajectory.append((state.copy() * player, action_probs.copy(), player))

        action = np.random.choice(env.action_size, p=action_probs)
        mcts.make_move(action)
        state = env.get_next_state(state, action, player)
        reward, is_terminal = env.get_value_and_terminated(
            state, action, player)
        if is_terminal:
            winning_player = player if reward > 0 else (
                env.get_opponent(player) if reward < 0 else 0)

            # back‑propagate the game outcome into the stored positions
            for i, (st, pol, ply) in enumerate(trajectory):
                outcome = 1 if ply == winning_player else (
                    -1 if winning_player != 0 else 0)
                trajectory[i] = (st, pol, outcome)
            return trajectory

        player = env.get_opponent(player)
