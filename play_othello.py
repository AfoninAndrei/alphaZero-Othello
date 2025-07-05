import argparse
import sys
import torch
import numpy as np
import pygame  # Needed for the pygame.time.wait() call.
from MCTS_model import MCTS
from envs.othello import OthelloGameNew as OthelloGame
from ui.OthelloUI import OthelloUI
from Models import FastOthelloNet, AlphaZeroNet


def choose_move(state, current_player, mcts_player, env, mcts, ui):
    """
    Choose a move given the current state and player.
      - For the computer (player 1), use MCTS.
      - For the human (player -1), use the UI to get input.
    """
    valid_moves = env.get_valid_moves(state, current_player)
    if current_player == mcts_player:
        action_probs = mcts.policy_improve_step(state,
                                                current_player,
                                                temp=0.0)
        # action_probs = mcts.inference(state, current_player)[0]
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


def play_human_vs_mcts(mcts_player: int, use_big_model: bool):
    # num_simulations controls level of bot: more is harder
    args = {'c_puct': 3.0, 'num_simulations': 1200}

    board_size = 8
    env = OthelloGame(board_size)
    if use_big_model:
        policy = AlphaZeroNet(board_size, 65, 5, 128)
        MODEL_PATH = 'othello_policy_RL_big.pt'
    else:
        policy = FastOthelloNet(board_size, 65)
        MODEL_PATH = 'othello_policy_RL_small.pt'

    policy.load_state_dict(torch.load(MODEL_PATH))
    policy.eval()

    mcts = MCTS(env, args, policy, apply_symmetry=True)

    # Initialize the UI.
    ui = OthelloUI(board_size)

    state = env.get_initial_state()
    current_player = 1

    running = True
    while running:
        valid_moves = env.get_valid_moves(state, current_player)
        ui.draw_board(state, valid_moves)
        ui.update_display()

        action = choose_move(state, current_player, mcts_player, env, mcts, ui)
        mcts.make_move(action)
        win_prob = (mcts.root.value + 1) / 2 if mcts.root else 0.0
        ui.set_win_prob(win_prob)

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


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Human vs. AlphaZero-style MCTS Othello bot")
    p.add_argument("--bot-player",
                   type=int,
                   choices=[1, -1],
                   default=1,
                   help="Colour the bot plays as (1 = black, -1 = white).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--big-model",
                   dest="use_big_model",
                   action="store_true",
                   help="Use the large AlphaZero network (default).")
    g.add_argument("--no-big-model",
                   dest="use_big_model",
                   action="store_false",
                   help="Use the small FastOthelloNet instead.")
    p.set_defaults(use_big_model=True)  # <-- default = True
    return p.parse_args()


if __name__ == "__main__":
    cli = parse_cli()
    play_human_vs_mcts(cli.bot_player, cli.use_big_model)
