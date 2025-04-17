import copy
import torch
import numpy as np

from MCTS_model import MCTS


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
        # print(state, action_probs, player)
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
