import torch
import numpy as np


@torch.no_grad()
def evaluate_models(env, mcts_a, mcts_b, n_matches=20):
    """
    Compare mcts_a vs mcts_b by playing n_matches in the given environment.
    Return fraction of wins by mcts_a.

    We'll alternate who starts first for fairness:
      - If current_player=1 => mcts_a is playing as +1
      - If current_player=-1 => mcts_a is playing as -1
    """
    wins_a = 0
    current_player = 1
    for _ in range(n_matches):
        result = play_match(env, mcts_a, mcts_b, current_player)
        # Alternate who goes first next game
        current_player *= -1
        if result == "A":
            wins_a += 1
    return wins_a / n_matches


def play_match(env, mcts_a, mcts_b, current_player):
    """
    Play a single game of (mcts_a) vs (mcts_b) in 'env'.
    'current_player' can be +1 or -1, indicating who moves first.

    Return "A" if mcts_a wins, "B" if mcts_b wins, or "Draw" if no winner.
    """
    state = env.get_initial_state()

    while True:
        valid_moves = env.get_valid_moves(state)
        if valid_moves.sum() == 0:
            return "Draw"  # no moves => draw

        if current_player == 1:
            # mcts_a picks move
            action_probs = mcts_a.policy_improve_step(
                init_state=state, init_player=current_player, temp=0.0)
        else:
            # mcts_b picks move
            action_probs = mcts_b.policy_improve_step(
                init_state=state, init_player=current_player, temp=0.0)

        # pick the highest-prob move
        action = np.argmax(action_probs)

        # step the environment
        state = env.get_next_state(state, action, current_player)
        reward, done = env.get_value_and_terminated(state, action)
        if done:
            # If reward=1 from the perspective of 'current_player'
            if reward == 1.0 and current_player == 1:
                return "A"
            elif reward == 1.0 and current_player == -1:
                return "B"
            else:
                return "Draw"
        # switch
        current_player = env.get_opponent(current_player)
