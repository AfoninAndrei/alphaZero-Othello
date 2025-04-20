import sys
from typing import Iterable, List, Tuple

import pygame
import numpy as np

# ----------------------------------------------------------------------------
#  Othello UI with animations, cleaner look, and board notations (v1.3)
# ----------------------------------------------------------------------------


class OthelloUI:
    """An animated UI for an N×N Othello board using Pygame.

    Changelog v1.3
    ----------------
    • **Board notations** – an extra header row and left‑hand column now show
      column letters (a, b, …) and row numbers (1, 2, …) so moves can be
      announced like **"4d"**.
    • Internal coordinates now account for a *label margin* equal to one cell
      size.
    • All public methods keep the same signature as v1.2.
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
        self.label_margin = cell_size  # one extra cell for labels (top & left)
        self.window_size = board_size * cell_size + self.label_margin
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

        # Font for labels
        self.font = pygame.font.SysFont(None, int(cell_size * 0.4))
        self.win_prob: float | None = None

    # ------------------------------------------------------------------
    #  Board drawing helpers
    # ------------------------------------------------------------------

    def set_win_prob(self, prob: float | None):
        """Store the latest win‑probability estimate (0–1)."""
        self.win_prob = prob

    def _draw_grid(self) -> None:
        """Draws the (N×N) grid and row/column labels."""
        self.screen.fill(self.BOARD_GREEN)

        off = self.label_margin
        board_px = self.board_size * self.cell_size

        # Main grid lines
        for i in range(self.board_size + 1):
            # Horizontal
            y = off + i * self.cell_size
            pygame.draw.line(self.screen, self.GRID_COLOR, (off, y),
                             (off + board_px, y), 3)
            # Vertical
            x = off + i * self.cell_size
            pygame.draw.line(self.screen, self.GRID_COLOR, (x, off),
                             (x, off + board_px), 3)

        # Column letters (top header row)
        for c in range(self.board_size):
            letter = chr(ord('a') + c)
            text_surf = self.font.render(letter, True, self.WHITE)
            text_rect = text_surf.get_rect()
            text_rect.center = (off + c * self.cell_size + self.cell_size // 2,
                                off // 2)
            self.screen.blit(text_surf, text_rect)

        # Row numbers (left‑hand column)
        for r in range(self.board_size):
            number = str(r + 1)
            text_surf = self.font.render(number, True, self.WHITE)
            text_rect = text_surf.get_rect()
            text_rect.center = (off // 2,
                                off + r * self.cell_size + self.cell_size // 2)
            self.screen.blit(text_surf, text_rect)

    def _draw_pieces(self, board: np.ndarray) -> None:
        radius = self.cell_size // 2 - 6
        off = self.label_margin
        for r in range(self.board_size):
            for c in range(self.board_size):
                val = board[r, c]
                if val == 0:
                    continue
                colour = self.BLACK if val == 1 else self.WHITE
                centre = (off + c * self.cell_size + self.cell_size // 2,
                          off + r * self.cell_size + self.cell_size // 2)
                pygame.draw.circle(self.screen, colour, centre, radius)

    def _draw_valid_moves(self, valid_moves: Iterable[int] | None) -> None:
        if valid_moves is None:
            return
        off = self.label_margin
        for idx, mv in enumerate(valid_moves):
            if not mv or idx >= self.board_size * self.board_size:
                continue
            r, c = divmod(idx, self.board_size)
            top_left = (off + c * self.cell_size + (self.cell_size - 20) // 2,
                        off + r * self.cell_size + (self.cell_size - 20) // 2)
            self.screen.blit(self._valid_dot, top_left)

    def _draw_value_estimate(self) -> None:
        if self.win_prob is None:
            return
        pct = f"{self.win_prob*100:5.1f}%"
        text = self.font.render(f"win: {pct}", True, self.WHITE)
        rect = text.get_rect()

        # -------- updated placement (8 px from top‑right corner) --------
        margin = 3
        rect.topleft = (margin, margin)
        # ----------------------------------------------------------------

        self.screen.blit(text, rect)

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
        self._draw_value_estimate()

    # ------------------------------------------------------------------
    #  Animation helpers
    # ------------------------------------------------------------------

    def _animate_piece_drop(self, pos: Tuple[int, int], colour: Tuple[int, int,
                                                                      int]):
        r, c = pos
        off = self.label_margin
        centre = (off + c * self.cell_size + self.cell_size // 2,
                  off + r * self.cell_size + self.cell_size // 2)
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
        off = self.label_margin
        for f in range(self.anim_frames):
            self._draw_grid()
            self.screen.blit(self._bg_buffer, (0, 0))
            for r, c in flips:
                centre = (off + c * self.cell_size + self.cell_size // 2,
                          off + r * self.cell_size + self.cell_size // 2)
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

    def _pixel_to_board(self, x: int, y: int) -> Tuple[int, int] | None:
        """Convert pixel coordinates to (row, col) or *None* if outside board."""
        off = self.label_margin
        if x < off or y < off:
            return None
        x -= off
        y -= off
        c = x // self.cell_size
        r = y // self.cell_size
        if c >= self.board_size or r >= self.board_size:
            return None
        return r, c

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
                    rc = self._pixel_to_board(x, y)
                    if rc is None:
                        continue
                    r, c = rc
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
