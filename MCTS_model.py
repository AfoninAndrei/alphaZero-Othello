import math
import numpy as np
from typing import Dict, Any, List

np.random.seed(888)

EPS = 1e-8


class Node:

    def __init__(self,
                 env: Any,
                 args: Dict[str, Any],
                 state: np.ndarray,
                 player: int,
                 action: int = None,
                 prior: float = 0.0,
                 parent: "Node" = None):
        self.env = env
        self.args = args
        self.state = state
        self.player = player
        self.action = action
        self.prior = prior

        # Monte Carlo estimates
        self.value_sum = 0.0
        self.visit_count = 0

        self.parent = parent
        # Children: action -> Node
        self.children: Dict[Any, Node] = {}

        # Valid actions from this state
        self.valid_mask = self.env.get_valid_moves(self.state)
        self.valid_actions = np.nonzero(self.valid_mask)[0]

        if self.action is not None:
            self.terminal_value, self.is_terminal = self.env.get_value_and_terminated(
                self.state, self.action, -player)
            # it is a terminal state, then the prev player actually won
            self.terminal_value = -abs(self.terminal_value)
        else:
            # For the root, there was no "last action".
            self.terminal_value = 0
            self.is_terminal = False

    @property
    def value(self) -> float:
        """Mean value so far."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def _get_ucb_score(self, action: int, prior: float) -> float:
        if action not in self.children:
            score = self.args['c_puct'] * prior * math.sqrt(self.visit_count +
                                                            EPS)
            return score

        child = self.children[action]
        score = - child.value + self.args['c_puct'] * child.prior * \
                        math.sqrt(self.visit_count + EPS) / (1 + child.visit_count)
        return score

    def update(self, value: float):
        self.visit_count += 1
        self.value_sum += value

    def expand(self, action: int, child_state: np.ndarray, child_prior: float):
        next_player = self.env.get_opponent(self.player)
        child_node = Node(env=self.env,
                          args=self.args,
                          state=child_state,
                          player=next_player,
                          action=action,
                          prior=child_prior,
                          parent=self)
        self.children[action] = child_node

    def backpropagate(self, value: float):
        current_node = self
        sign = 1
        while current_node is not None:
            current_node.update(sign * value)
            # store everything from the root player's perspective
            # actions from even steps should not be maximised in values
            # since these are the actions of the opponent
            sign = -sign
            current_node = current_node.parent


class MCTS:

    def __init__(self, env, args, policy, use_rollout=False):
        self.env = env
        self.args = args
        self.policy = policy
        self.use_rollout = use_rollout
        self.num_actions = self.env.action_size
        # we need policy if not rollout
        if not self.use_rollout:
            assert self.policy is not None
        # Keep a persistent tree (root) between moves.
        self.root = None

    def make_move(self, action):
        # this assumes that env is deterministic
        if not self.root:
            # in case we play 2nd and it is the 1st move
            return
        self.root = self.root.children[action]
        # detach from the above tree
        self.root.parent = None

    def policy_improve_step(self,
                            init_state,
                            init_player: int,
                            temp=1) -> List[float]:
        # start the tree from the current state
        # we need to reuse already computed statistics
        if self.root is None:
            self.root = Node(env=self.env,
                             args=self.args,
                             state=init_state.copy(),
                             player=init_player,
                             action=None)
        else:
            # assrt if there is a descepancy between game state and tree state
            assert np.all(self.root.state == init_state)
            assert self.root.player == init_player

        for _ in range(self.args['num_simulations']):
            # simulate until the leaf node
            self._simulate(self.root)

        counts = np.zeros(self.num_actions, dtype=np.float32)
        for action, child_node in self.root.children.items():
            counts[action] = child_node.visit_count

        if abs(temp) < 1e-1:
            # Choose the action(s) with the highest visit_count
            best_actions = np.where(counts == counts.max())[0]
            best_action = np.random.choice(best_actions)
            probs = np.zeros_like(counts)
            probs[best_action] = 1.0

        else:
            counts_exp = counts**(1.0 / temp)
            norm = np.sum(counts_exp)
            if norm < 1e-12:
                # If everything is zero, pick uniformly from valid actions
                valid_actions = self.root.valid_actions
                if len(valid_actions) == 0:
                    # Terminal node => no moves
                    probs = np.zeros(self.num_actions, dtype=np.float32)
                else:
                    probs = np.zeros(self.num_actions, dtype=np.float32)
                    for a in valid_actions:
                        probs[a] = 1.0 / len(valid_actions)
            else:
                probs = counts_exp / norm

        # get a reference policy from the MCTS
        return probs

    def _rollout(self, state: np.ndarray, player: int) -> float:
        """
        Do a random simulation from (state, player) until the game ends.
        Return +1 if 'player' eventually wins, -1 if the opponent wins, or 0 if draw.
        """
        # Rollout is MC estimation of the the value function
        # Using policy for value setimation is like TD-learning
        current_state = state.copy()
        current_player = player
        while True:
            valid_moves = self.env.get_valid_moves(current_state)
            possible_actions = np.where(valid_moves == 1)[0]
            if len(possible_actions) == 0:
                # board is full => draw
                return 0.0

            action = np.random.choice(possible_actions)
            current_state = self.env.get_next_state(current_state, action,
                                                    current_player)
            terminal_value, is_terminal = self.env.get_value_and_terminated(
                current_state, action, current_player)

            if is_terminal:
                winner = np.sign(terminal_value)
                return abs(terminal_value
                           ) if winner == player else -abs(terminal_value)

            current_player = self.env.get_opponent(current_player)

    def _expand_and_evaluate(self, leaf: Node):
        """
        Use the policy to get (prior probabilities, value) at the leaf.
        Then expand *all valid actions* (or a subset if you prefer).
        Finally, backprop the value to the leaf.
        """
        if self.use_rollout:
            # uniform prior and value from MC estimation
            priors = np.ones(self.num_actions, dtype=np.float32)
            leaf_value = self._rollout(leaf.state, leaf.player)
        else:
            priors, leaf_value = self.policy.inference(leaf.state)

        # Zero out invalid moves
        priors *= leaf.valid_mask
        sum_priors = priors.sum()
        if sum_priors > 1e-12:
            priors /= sum_priors  # normalize

        # Expand each valid move
        for action in leaf.valid_actions:
            # Construct next state
            next_state = leaf.state.copy()
            # Apply the move for 'leaf.player'
            next_state = self.env.get_next_state(next_state, action,
                                                 leaf.player)

            # Child’s prior from policy distribution
            leaf.expand(action, next_state, priors[action])

        # Backprop the leaf_value from the policy to this leaf
        leaf.backpropagate(leaf_value)

    def _select_child(self, node: Node) -> Node:
        """
        PUCT selection among the children of 'node'.
        """
        best_score = -float('inf')
        best_action = None
        for action in node.valid_actions:
            if action not in node.children:
                continue

            prior = node.children[action].prior
            score = node._get_ucb_score(action, prior)
            if score > best_score:
                best_score = score
                best_action = action

        return node.children[best_action]

    def _simulate(self, root_node: Node):
        node = root_node

        # 1) Traverse down
        while True:
            if node.is_terminal:
                # Terminal leaf => backprop the known terminal value
                node.backpropagate(node.terminal_value)
                return
            if node.is_leaf():
                # Leaf => Expand it, then backprop
                self._expand_and_evaluate(node)
                return

            # Otherwise, pick a child to go deeper
            node = self._select_child(node)
