import pygame
import sys


class OthelloUI:

    def __init__(self, board_size, cell_size=80):
        self.board_size = board_size
        self.cell_size = cell_size
        self.window_size = board_size * cell_size
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.window_size, self.window_size))
        pygame.display.set_caption("Othello: Human vs MCTS")
        self.clock = pygame.time.Clock()

    def draw_board(self, board, valid_moves=None):
        GREEN = (34, 139, 34)
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GRAY = (200, 200, 200)
        self.screen.fill(GREEN)
        # Draw grid lines.
        for i in range(self.board_size + 1):
            pygame.draw.line(self.screen, BLACK, (0, i * self.cell_size),
                             (self.window_size, i * self.cell_size), 2)
            pygame.draw.line(self.screen, BLACK, (i * self.cell_size, 0),
                             (i * self.cell_size, self.window_size), 2)

        # Draw pieces.
        for row in range(self.board_size):
            for col in range(self.board_size):
                center = (col * self.cell_size + self.cell_size // 2,
                          row * self.cell_size + self.cell_size // 2)
                if board[row, col] == 1:
                    pygame.draw.circle(self.screen, WHITE, center,
                                       self.cell_size // 2 - 5)
                elif board[row, col] == -1:
                    pygame.draw.circle(self.screen, BLACK, center,
                                       self.cell_size // 2 - 5)

        # Optionally highlight valid moves.
        if valid_moves is not None:
            for idx, valid in enumerate(valid_moves):
                if valid:
                    # For board moves, highlight the cell.
                    if idx < self.board_size * self.board_size:
                        row = idx // self.board_size
                        col = idx % self.board_size
                        center = (col * self.cell_size + self.cell_size // 2,
                                  row * self.cell_size + self.cell_size // 2)
                        pygame.draw.circle(self.screen, GRAY, center, 10)
                    else:
                        # For the pass move, draw a "P" in the top-right corner.
                        font = pygame.font.SysFont(None, 48)
                        text = font.render("P", True, GRAY)
                        self.screen.blit(text, (self.window_size - 50, 10))

    def get_human_move(self, valid_moves):
        """
        Wait for the human user to either click on a valid board cell
        or press the 'p' key to pass (if pass is valid).
        Returns the action index.
        """
        move_selected = False
        action = None
        while not move_selected:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        # Pass move is at index board_size * board_size.
                        if valid_moves[self.board_size * self.board_size]:
                            action = self.board_size * self.board_size
                            move_selected = True
                            break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    col = pos[0] // self.cell_size
                    row = pos[1] // self.cell_size
                    action = row * self.board_size + col
                    if valid_moves[action]:
                        move_selected = True
                        break
            self.clock.tick(30)
        return action

    def update_display(self):
        pygame.display.flip()
