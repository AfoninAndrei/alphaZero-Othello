import sys
import numpy as np
import pygame  # Needed for the pygame.time.wait() call.
from MCTS_model import MCTS
from envs.othello import OthelloGame
from ui.OthelloUI import OthelloUI


def choose_move(state, current_player, env, mcts, ui):
    """
    Choose a move given the current state and player.
      - For the computer (player 1), use MCTS.
      - For the human (player -1), use the UI to get input.
    """
    valid_moves = env.get_valid_moves(state, current_player)
    if current_player == 1:
        action_probs = mcts.policy_improve_step(state,
                                                current_player,
                                                temp=0.0)
        action = int(np.argmax(action_probs))
        if action == env.action_size - 1:
            print("Computer chooses: pass")
        else:
            row = action // env.n
            col = action % env.n
            print("Computer chooses:", (row, col))
        return action
    else:
        print(
            "Your turn (player -1). Click on a valid cell or press 'P' to pass."
        )
        return ui.get_human_move(valid_moves)


def play_human_vs_mcts(policy):
    # Parameters for MCTS.
    args = {'c_puct': 1.0, 'num_simulations': 100, 'mcts_temperature': 1.0}
    board_size = 5  # 5x5 board.
    env = OthelloGame(board_size)
    # For computer moves, we use an MCTS instance.
    mcts = MCTS(env, args, None, True)

    # Initialize the UI.
    ui = OthelloUI(board_size)

    state = env.get_initial_state()  # shape: (board_size, board_size)
    # Let computer be player 1 and human be player -1.
    current_player = 1

    running = True
    while running:
        valid_moves = env.get_valid_moves(state, current_player)
        ui.draw_board(state, valid_moves)
        ui.update_display()

        action = choose_move(state, current_player, env, mcts, ui)
        mcts.make_move(action)
        state = env.get_next_state(state, action, current_player)
        value, done = env.get_value_and_terminated(state, action,
                                                   current_player)

        if done:
            ui.draw_board(state)
            ui.update_display()
            if value == 1:
                print("Game Over! Winner: Computer (player 1)")
            elif value == -1:
                print("Game Over! Winner: Human (player -1)")
            else:
                print("Game Over! Draw")
            pygame.time.wait(3000)
            running = False
        current_player = env.get_opponent(current_player)

    ui.screen.fill((0, 0, 0))
    ui.update_display()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    # Replace None with your actual policy instance if available.
    play_human_vs_mcts(None)
