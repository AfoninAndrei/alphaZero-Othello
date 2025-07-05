#!/usr/bin/env python3
"""
batch_vs_egaroucid.py – run many head-less games versus Egaroucid

  python batch_vs_egaroucid.py        # 10 games (default)
  python batch_vs_egaroucid.py 50     # 50 games
"""
import re
import os, sys, queue, subprocess, threading, time, random
from pathlib import Path

import numpy as np
import torch

from MCTS_model import MCTS
from envs.othello import OthelloGameNew as OthelloGame
from Models import FastOthelloNet, AlphaZeroNet
# model 15 is best so far
# ────────── paths & parameters you may want to change ────────── #
ENGINE_PATH = Path(
    "/Users/andreiafonin/Downloads/Egaroucid/bin/Egaroucid_for_Console.out")

MODEL_PATH = "othello_policy_RL_big_best_cluster15.pt"
# MODEL_PATH = "othello_policy_RL_fast_best_cluster7.pt"
ENGINE_CWD = ENGINE_PATH.parent  # ← contains resources/ as sub-dir
ENGINE_LEVEL = 10  # 1 = fast … 60 = strong
N_GAMES = 50
MCTS_SIMS = 1200  # ↓ for speed, ↑ for strength
C_PUCT = 3.0

APPLY_SYMMETRY = True
TIMEOUT = 10  # seconds to wait for any reply
# ──────────────────────────────────────────────────────────────── #

# board helpers --------------------------------------------------
_COL2 = "abcdefgh"
_C2I = {c: i for i, c in enumerate(_COL2)}


def rc2sq(r, c):
    return f"{_COL2[c]}{r+1}"


def sq2a(s, n=8):
    return n * n if s == "pass" else (int(s[1]) - 1) * n + _C2I[s[0]]


# ──────────────── tiny NBoard wrapper (prompt fix) ──────────────
MOVE_PAT = re.compile(r"\b([a-h][1-8]|pass)\b", re.I)

import re, queue, subprocess, threading, time
from pathlib import Path

# ---------- customise these two paths once ----------
ENGINE_PATH = Path(
    "/Users/andreiafonin/Downloads/Egaroucid/bin/Egaroucid_for_Console.out")
ENGINE_CWD = ENGINE_PATH.parent  # ← contains *resources/* sub-dir
# -----------------------------------------------------
TIMEOUT = 10  # seconds to wait for any reply
MOVE_RX = re.compile(r"\b([a-h][1-8]|pass)\b", re.I)
MOVE_TABLE_RX = re.compile(
    r"\|\s*(?:Book|-|\d+)\|\s*(?:-|\d+)\|\s*([a-h][1-8]|pass)\s*\|", re.I)
PROMPT_RX = re.compile(r"^\s*8\s")


class NBEngine:
    """Minimal, fool-proof wrapper around *Egaroucid -nboard*."""

    PROMPT = "8"  # always the last line of any reply

    # ───────────────────────── start-up ──────────────────────────
    def __init__(self, level: int = 5):
        self.proc = subprocess.Popen(
            [str(ENGINE_PATH), "-nboard", "-level",
             str(level)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge so we never miss errors
            text=True,
            bufsize=1,
            cwd=ENGINE_CWD)
        self.q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

        # engine prints nothing until it gets a first command → send “pass”
        self._cmd_wait("play pass")

    # ───────────────────── public commands ──────────────────────
    def play(self, move: str):
        """`move` is like 'd3' … 'h8' or 'pass' (no colour prefix)."""
        self._cmd_wait(f"play {move}")

    def genmove(self):
        """
        Sends ‘go’, waits until the prompt returns, then extracts the move
        **from the search-table line**.  Falls back to the old “take the
        last coordinate” strategy if that line is missing.
        Returns        (move:str, game_over:bool)
        """
        block = self._cmd_wait("go")

        move = None
        # 1️⃣  First try the dedicated Move column
        for ln in block:
            m = MOVE_TABLE_RX.search(ln)
            if m:
                move = m.group(1).lower()
                break

        # 2️⃣  Fallback – last legal coordinate in the whole block
        if move is None:
            for ln in reversed(block):
                m_iter = list(MOVE_RX.finditer(ln))
                if m_iter:
                    move = m_iter[-1].group(1).lower()
                    break

        if move is None:
            raise RuntimeError("‘go’ finished but no move found:\n" +
                               "\n".join(block))

        game_over = any("game over" in ln.lower() for ln in block)
        # print("Extracted move", move)
        return move, game_over

    def quit(self):
        """
        Clean shutdown: try 'quit' once, but don’t block if the engine has
        already closed its end of the pipe.  Always call .terminate() as
        the last step so the OS reaps the child whatever happened.
        """
        try:
            # send 'quit' only if stdin is still open
            if self.proc.stdin and self.proc.poll() is None:
                try:
                    self.proc.stdin.write("quit\n")
                    self.proc.stdin.flush()
                except BrokenPipeError:
                    pass  # engine already exited
            # give it a very short moment to exit cleanly
            self.proc.wait(timeout=0.2)
        except (subprocess.TimeoutExpired, OSError):
            pass  # ignore – we'll kill it next
        finally:
            self.proc.terminate()

    # ─────────────────────── internals ──────────────────────────
    def _pump(self):
        for ln in self.proc.stdout:  # background reader thread
            self.q.put(ln.rstrip("\n"))

    def _get(self):
        return self.q.get(timeout=TIMEOUT)  # raises TimeoutError if silent

    def _cmd_wait(self, txt: str):
        """
        Send `txt`, then collect **all** lines until the bottom row (“8 …”)
        appears.  Returns the collected list (the prompt line itself is
        discarded).
        """
        # clear any stray lines still in the queue – keeps parsing in sync
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

        self.proc.stdin.write(txt + "\n")
        self.proc.stdin.flush()

        lines = []
        while True:
            ln = self._get()  # may raise TimeoutError
            if PROMPT_RX.match(ln):
                # print("\n".join(lines))
                return lines  # prompt → command finished
            lines.append(ln)


# ───────────── convenience wrappers for your MCTS agent ─────────
def new_mcts(env):
    # net = FastOthelloNet(env.n, env.action_size)
    net = AlphaZeroNet(env.n, env.action_size, 5, 128)  # create model instance
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()
    return MCTS(env, {
        'c_puct': C_PUCT,
        'num_simulations': MCTS_SIMS
    },
                net,
                apply_symmetry=APPLY_SYMMETRY)


def mcts_move(env, mcts, st, pl):
    p = mcts.policy_improve_step(st, pl, temp=0.0)
    return int(np.argmax(p))


# ───────────────────── play ONE complete game ───────────────────
def play_one(env, black_is_mcts):
    eng = NBEngine(ENGINE_LEVEL)

    st = env.get_initial_state()
    pl = 1  # 1 = black
    mcts = new_mcts(env)
    engine_final = False

    while True:
        mcts_turn = (pl == 1) == black_is_mcts
        if mcts_turn:
            a = mcts_move(env, mcts, st, pl)
            if a == env.action_size - 1:
                eng.play("pass")
            else:
                r, c = divmod(a, env.n)
                eng.play(rc2sq(r, c))
        else:
            valid = env.get_valid_moves(st, pl)
            if valid[:-1].sum() == 0:  # ⬅︎ NEW  – only ‘pass’ legal
                # engine must pass – keep boards in sync but skip genmove()
                eng.play("pass")
                a = env.action_size - 1
            else:
                mv, game_over = eng.genmove()
                a = env.action_size - 1 if mv == "pass" else sq2a(mv, env.n)
                if game_over:
                    engine_final = True

        st = env.get_next_state(st, a, pl)
        mcts.make_move(a)

        val, done = env.get_value_and_terminated(st, a, pl)
        if done or engine_final:
            eng.quit()
            env.print_board(st, player=pl)
            return val * (1 if pl == 1 else -1)  # >0  → Black wins

        pl = env.get_opponent(pl)


# ───────────────────────── batch driver ─────────────────────────
def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_GAMES
    env = OthelloGame(8)
    res = {"win": 0, "loss": 0, "draw": 0}

    for g in range(1, n + 1):
        black_is_mcts = (g % 2) == 1
        print(
            f"Game {g:>3}/{n} – MCTS is {'Black' if black_is_mcts else 'White'} …",
            end=" ",
            flush=True)
        out = play_one(env, black_is_mcts)
        if out > 0:
            res["win" if black_is_mcts else "loss"] += 1
            winner = "MCTS" if black_is_mcts else "Egaroucid"
        elif out < 0:
            res["loss" if black_is_mcts else "win"] += 1
            winner = "Egaroucid" if black_is_mcts else "MCTS"
        else:
            res["draw"] += 1
            winner = "Draw"

        print(f"{winner} wins.\n", flush=True)

    print("\nfinal tally")
    print(" wins :", res["win"])
    print(" loss :", res["loss"])
    print(" draw :", res["draw"])
    print(" score:", res["win"] - res["loss"])


if __name__ == "__main__":
    main()
