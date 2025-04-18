import sys
from typing import Iterable, List, Tuple

import pygame
import numpy as np

# ----------------------------------------------------------------------------
#  Othello UI with animations and a cleaner look (v1.2 – slower animations)
# ----------------------------------------------------------------------------


class OthelloUI:
    """A lightweight, animated UI for an N×N Othello board using Pygame.

    Changelog v1.2
    --------------
    • **Slower animations** – piece drop & flips now run over **600 ms** instead
      of 300 ms and a short 250 ms pause is inserted after the move completes so
      players can register where the disc was placed.
    • Version bump only; public API unchanged.
    """

    # ------------------------------------------------------------------
    #  Construction & basic colours
    # ------------------------------------------------------------------

    BOARD_GREEN: Tuple[int, int, int] = (30, 120, 50)
    GRID_COLOR: Tuple[int, int, int] = (20, 60, 30)
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    WHITE: Tuple[int, int, int] = (245, 245, 245)

    GRAY: Tuple[int, int, int] = (180, 180, 180)
    VALID_DOT_RGBA: Tuple[int, int, int,
                          int] = (*GRAY, 160)  # semi‑transparent

    # How long each animation lasts (seconds)
    ANIM_DURATION_S = 0.6  # 600 ms
    POST_MOVE_PAUSE_MS = 250  # pause after entire move (ms)

    def __init__(self,
                 board_size: int,
                 cell_size: int = 80,
                 anim_fps: int = 60):
        self.board_size = board_size
        self.cell_size = cell_size
        self.window_size = board_size * cell_size
        self.anim_fps = anim_fps
        self.anim_frames = int(self.ANIM_DURATION_S * anim_fps)

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.window_size, self.window_size))
        pygame.display.set_caption("Othello: Human vs MCTS")
        self.clock = pygame.time.Clock()

        # Pre‑build the semi‑transparent dot used to mark valid moves
        self._valid_dot = pygame.Surface((20, 20), flags=pygame.SRCALPHA)
        pygame.draw.circle(self._valid_dot, self.VALID_DOT_RGBA, (10, 10), 8)

    # ------------------------------------------------------------------
    #  Board drawing helpers
    # ------------------------------------------------------------------

    def _draw_grid(self) -> None:
        """Draws the N×N grid on the current screen surface."""
        self.screen.fill(self.BOARD_GREEN)
        for i in range(self.board_size + 1):
            # Horizontal lines
            y = i * self.cell_size
            pygame.draw.line(self.screen, self.GRID_COLOR, (0, y),
                             (self.window_size, y), 3)
            # Vertical lines
            x = i * self.cell_size
            pygame.draw.line(self.screen, self.GRID_COLOR, (x, 0),
                             (x, self.window_size), 3)

    def _draw_pieces(self, board: np.ndarray) -> None:
        radius = self.cell_size // 2 - 6
        for r in range(self.board_size):
            for c in range(self.board_size):
                val = board[r, c]
                if val == 0:
                    continue
                colour = self.BLACK if val == 1 else self.WHITE
                centre = (c * self.cell_size + self.cell_size // 2,
                          r * self.cell_size + self.cell_size // 2)
                pygame.draw.circle(self.screen, colour, centre, radius)

    def _draw_valid_moves(self, valid_moves: Iterable[int] | None) -> None:
        if valid_moves is None:
            return
        for idx, mv in enumerate(valid_moves):
            if not mv or idx >= self.board_size * self.board_size:
                continue
            r, c = divmod(idx, self.board_size)
            top_left = (c * self.cell_size + (self.cell_size - 20) // 2,
                        r * self.cell_size + (self.cell_size - 20) // 2)
            self.screen.blit(self._valid_dot, top_left)

    # ------------------------------------------------------------------
    #  Public drawing API
    # ------------------------------------------------------------------

    def draw_board(self,
                   board: np.ndarray,
                   valid_moves: Iterable[int] | None = None) -> None:
        """Re‑render *board* immediately. Call `update_display()` afterwards."""
        self._draw_grid()
        self._draw_pieces(board)
        self._draw_valid_moves(valid_moves)

    # ------------------------------------------------------------------
    #  Animation helpers
    # ------------------------------------------------------------------

    def _animate_piece_drop(self, pos: Tuple[int, int], colour: Tuple[int, int,
                                                                      int]):
        r, c = pos
        centre = (c * self.cell_size + self.cell_size // 2,
                  r * self.cell_size + self.cell_size // 2)
        max_radius = self.cell_size // 2 - 6
        for f in range(self.anim_frames):
            self._draw_grid()
            self.screen.blit(self._bg_buffer, (0, 0))
            radius = int(max_radius * (f + 1) / self.anim_frames)
            pygame.draw.circle(self.screen, colour, centre, radius)
            pygame.display.flip()
            self.clock.tick(self.anim_fps)

    def _animate_flips(self, flips: List[Tuple[int, int]],
                       old_colour: Tuple[int, int, int],
                       new_colour: Tuple[int, int, int]):
        max_radius = self.cell_size // 2 - 6
        half = self.anim_frames // 2
        for f in range(self.anim_frames):
            self._draw_grid()
            self.screen.blit(self._bg_buffer, (0, 0))
            for r, c in flips:
                centre = (c * self.cell_size + self.cell_size // 2,
                          r * self.cell_size + self.cell_size // 2)
                if f < half:
                    # shrink old colour
                    radius = int(max_radius * (1 - f / half))
                    colour = old_colour
                else:
                    # grow new colour
                    radius = int(max_radius * ((f - half) / half))
                    colour = new_colour
                radius = max(radius, 2)
                pygame.draw.circle(self.screen, colour, centre, radius)
            pygame.display.flip()
            self.clock.tick(self.anim_fps)

    # ------------------------------------------------------------------
    #  Animation entry point
    # ------------------------------------------------------------------

    def animate_move(self, prev_board: np.ndarray, next_board: np.ndarray):
        """Animate the transition from *prev_board* to *next_board*."""
        # Buffer background (grid + previous pieces) once
        self.draw_board(prev_board)
        self.update_display()
        self._bg_buffer = self.screen.copy()

        # Detect new piece position (prev==0 & next!=0) and flips (prev!=next)
        mask_new_piece = (prev_board == 0) & (next_board != 0)
        new_positions = np.argwhere(mask_new_piece)
        new_piece_pos = tuple(new_positions[0]) if len(new_positions) else None

        flips_pos: List[Tuple[int, int]] = list(
            map(tuple, np.argwhere(prev_board != next_board)))
        if new_piece_pos and new_piece_pos in flips_pos:
            flips_pos.remove(new_piece_pos)

        # Animate drop
        if new_piece_pos:
            colour = self.BLACK if next_board[
                new_piece_pos] == 1 else self.WHITE
            self._animate_piece_drop(new_piece_pos, colour)
        else:
            colour = self.BLACK  # fallback, won't be used if no flips

        # Animate flips
        if flips_pos:
            old_colour = self.WHITE if colour == self.BLACK else self.BLACK
            self._animate_flips(flips_pos, old_colour, colour)

        # Final state & small pause so players can see the board
        self.draw_board(next_board)
        self.update_display()
        pygame.time.delay(self.POST_MOVE_PAUSE_MS)

    # ------------------------------------------------------------------
    #  User input
    # ------------------------------------------------------------------

    def get_human_move(self, valid_moves: Iterable[int]):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    pass_idx = self.board_size * self.board_size
                    if valid_moves[pass_idx]:
                        return pass_idx
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    c, r = x // self.cell_size, y // self.cell_size
                    idx = r * self.board_size + c
                    if valid_moves[idx]:
                        return idx
            self.clock.tick(30)

    # ------------------------------------------------------------------
    #  Display helper
    # ------------------------------------------------------------------

    @staticmethod
    def update_display():
        pygame.display.flip()
