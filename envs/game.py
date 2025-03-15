# games/game.py
from abc import ABC, abstractmethod, abstractproperty


class Game(ABC):

    @abstractmethod
    def get_initial_state(self):
        """
        Return the initial state of the board.
        """
        pass

    @abstractmethod
    def get_valid_moves(self, state, player):
        """
        Given a state and current player, return a binary vector of valid moves.
        """
        pass

    @abstractproperty
    def action_size(self):
        """
        Returns number of possible actions
        """
        pass

    @abstractproperty
    def state_size(self):
        """
        Returns number of states
        """
        pass

    @abstractmethod
    def get_next_state(self, state, action, player):
        """
        Given a state, an action, and the current player,
        return the next state.
        """
        pass

    @abstractmethod
    def get_value_and_terminated(self, state, action, player):
        """
        Given a state, executed action and the current player, return:
            - a value: 1 if player wins, -1 if loses, 0 for draw or ongoing.
            - a boolean indicating whether the game is over.
        """
        pass

    @abstractmethod
    def get_opponent(self, player):
        """
        Return the opponent of the current player (usually just -player)
        """
        pass
