import torch
import numpy as np

from MCTS_model import MCTS
from envs.othello import OthelloGameNew as OthelloGame

# def get_training_data(trajectory, winning_player):
#     for i, (st, pol, ply, value) in enumerate(trajectory):
#         outcome = 1 if ply == winning_player else (
#             -1 if winning_player != 0 else 0)
#         trajectory[i] = (st, pol, outcome)
#     return trajectory


def get_training_data(trajectory, winning_player, lambd: float = 0.9):

    def z_for(p):
        if winning_player == 0:  # draw
            return 0.0
        return 1.0 if p == winning_player else -1.0

    out = [None] * len(trajectory)
    G_next = None  # G_{t+1}
    next_player = None  # player_{t+1}

    # walk the trajectory backward
    for t in reversed(range(len(trajectory))):
        state, π, player, v_root = trajectory[t]
        mc = z_for(player)  # Monte‑Carlo outcome from *this* player’s view

        if G_next is None:  # closest to terminal
            G_t = mc
        else:
            # flip sign if the side‑to‑move switched between t and t+1
            sign = 1.0 if player == next_player else -1.0
            G_t = (1.0 - lambd) * v_root + lambd * sign * G_next

        out[t] = (state, π, G_t)
        G_next = G_t
        next_player = player  # becomes “player_{t+1}” for the previous step

    return out


@torch.no_grad()
def one_self_play(args_tuple):
    """
    Single complete self‑play game executed in a child process.
    Returns a list of (state, improved_policy, value) triples.
    """
    board_size, args, policy_state, inference_cache = args_tuple

    # ----- reconstruct lightweight objects inside the worker
    env = OthelloGame(board_size)
    (policy_class, policy_config, policy_state_dict) = policy_state
    policy = policy_class(**policy_config)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    mcts = MCTS(env,
                args,
                policy,
                dirichlet_alpha=args["dirichlet_alpha"],
                dirichlet_epsilon=args["dirichlet_epsilon"],
                inference_cache=inference_cache)

    trajectory = []
    state = env.get_initial_state()
    player, is_terminal = 1, False

    while not is_terminal:
        # first n moves we allow more randomness similar to AlphaZero
        temperature = args["mcts_temperature"] if len(
            trajectory) < args["num_exploratory_moves"] else 0.0
        action_probs = mcts.policy_improve_step(state,
                                                player,
                                                temp=temperature)

        trajectory.append((state.copy() * player, action_probs.copy(), player,
                           mcts.root.value))

        action = np.random.choice(env.action_size, p=action_probs)
        mcts.make_move(action)
        state = env.get_next_state(state, action, player)
        reward, is_terminal = env.get_value_and_terminated(
            state, action, player)
        if is_terminal:
            winning_player = player if reward > 0 else (
                env.get_opponent(player) if reward < 0 else 0)

            # back‑propagate the game outcome into the stored positions
            return get_training_data(trajectory, winning_player)

        player = env.get_opponent(player)
