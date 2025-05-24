import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import math
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import Dict, Any, List

EPS = 1e-8
VIRTUAL_LOSS = 1.0


class Node:

    __slots__ = (
        "env",
        "args",
        "state",
        "player",
        "action",
        "prior",
        "value_sum",
        "visit_count",
        "virtual_visits",
        "virtual_value",
        "parent",
        "children",
        "valid_mask",
        "valid_actions",
        "terminal_value",
        "is_terminal",
        "lock",
    )

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

        # Virtual loss bookkeeping for multithreading
        self.virtual_visits = 0
        self.virtual_value = 0.0

        self.parent = parent
        # Children: action -> Node
        self.children: Dict[Any, Node] = {}

        # Valid actions from this state
        self.valid_mask = self.env.get_valid_moves(self.state, player)
        self.valid_actions = np.nonzero(self.valid_mask)[0]

        if self.action is not None:
            # Previous player actually made last move
            self.terminal_value, self.is_terminal = self.env.get_value_and_terminated(
                self.state, self.action, player)
        else:
            # For the root, there was no "last action".
            self.terminal_value = 0
            self.is_terminal = False

        self.lock = threading.Lock()

    @property
    def value(self) -> float:
        total_v = self.value_sum + self.virtual_value
        total_n = self.visit_count + self.virtual_visits
        return 0.0 if total_n == 0 else total_v / total_n

    def add_virtual_loss(self):
        with self.lock:
            self.virtual_visits += 1
            self.virtual_value += VIRTUAL_LOSS

    def revert_virtual_loss(self):
        with self.lock:
            self.virtual_visits -= 1
            self.virtual_value -= VIRTUAL_LOSS

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def _get_ucb_score(self, action: int, prior: float) -> float:
        if action not in self.children:
            return self.args["c_puct"] * prior * math.sqrt(
                self.visit_count + self.virtual_visits + EPS)

        child = self.children[action]
        q = -child.value  # from parent player's perspective
        u = (self.args["c_puct"] * child.prior *
             math.sqrt(self.visit_count + self.virtual_visits + EPS) /
             (1 + child.visit_count + child.virtual_visits))
        return q + u

    def update(self, value: float):
        with self.lock:
            self.visit_count += 1
            self.value_sum += value

    def expand(self, action: int, child_prior: float):
        action = int(action)
        next_player = self.env.get_opponent(self.player)
        child_state = self.env.get_next_state(self.state, action, self.player)
        child_node = Node(env=self.env,
                          args=self.args,
                          state=child_state,
                          player=next_player,
                          action=action,
                          prior=child_prior,
                          parent=self)
        with self.lock:
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

    def __init__(self,
                 env,
                 args,
                 policy,
                 dirichlet_alpha=0.03,
                 dirichlet_epsilon=0.0,
                 inference_cache=None):
        self.env = env
        self.args = args
        self.policy = policy
        self.use_rollout = False
        self.num_actions = self.env.action_size
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        # we need policy if not rollout
        if self.policy is None:
            self.use_rollout = True

        # Keep a persistent tree (root) between moves.
        self.root = None
        self.num_threads: int = max(1, self.args.get("num_threads", 4))
        self.pool = ThreadPoolExecutor(max_workers=self.num_threads)
        self.inference_cache = inference_cache

    def make_move(self, action):
        # this assumes that env is deterministic
        action = int(action)
        # for _, child in self.root.children.items():
        #     print(child.action, child.visit_count, child.value, child.prior)

        # print("Chosen:", self.root.children[action].action,
        #       self.root.children[action].value)  # value stored on edge

        if not self.root:
            # in case we play 2nd and it is the 1st move
            return

        # detach from the above tree
        self.root = self.root.children[action]
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

        if self.root.is_leaf():  # root never explored yet
            self._expand_and_evaluate(self.root)

        sims = self.args["num_simulations"]
        futs = [
            self.pool.submit(self._simulate, self.root) for _ in range(sims)
        ]
        for f in futs:
            f.result()

        counts = np.zeros(self.num_actions, dtype=np.float32)
        for action, child_node in self.root.children.items():
            # vist count is reused from prev simulations
            counts[action] = child_node.visit_count

        if abs(temp) < 1e-1:
            # Choose the action(s) with the highest visit_count
            # COUNT_NUMBER = 70
            # if np.any(counts > COUNT_NUMBER):
            #     best_value = float('inf')
            #     for action, child_node in self.root.children.items():
            #         if child_node.visit_count > COUNT_NUMBER and child_node.value < best_value:
            #             best_action = action
            #             best_value = child_node.value
            # else:
            best_actions = np.where(counts == counts.max())[0]
            best_action = np.random.choice(best_actions)
            probs = np.zeros_like(counts)

            if len(self.root.valid_actions) != 0:
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
            valid_moves = self.env.get_valid_moves(current_state,
                                                   current_player)
            possible_actions = np.where(valid_moves == 1)[0]
            if len(possible_actions) == 0:
                # board is full => draw
                return 0.0

            action = np.random.choice(possible_actions)
            current_state = self.env.get_next_state(current_state, action,
                                                    current_player)
            # we need to eval from the initial player perspective
            terminal_value, is_terminal = self.env.get_value_and_terminated(
                current_state, action, player)

            if is_terminal:
                return terminal_value

            current_player = self.env.get_opponent(current_player)

    def _policy_inference(self, leaf):
        if not self.inference_cache:
            return self.policy.inference(leaf.state, leaf.player)

        key = (leaf.state, leaf.player)
        if key in self.inference_cache:
            return self.inference_cache[key]

        self.inference_cache[key] = self.policy.inference(
            leaf.state, leaf.player)
        return self.inference_cache[key]

    def _expand_and_evaluate(self, leaf: Node):
        """
        Use the policy to get (prior probabilities, value) at the leaf.
        Then expand *all valid actions* (or a subset if you prefer).
        Finally, backprop the value to the leaf.
        """
        # get leaf value from the leaf player perspective
        if self.use_rollout:
            # uniform prior and value from MC estimation
            priors = np.ones(self.num_actions, dtype=np.float32)
            leaf_value = self._rollout(leaf.state, leaf.player)
        else:
            priors, leaf_value = self._policy_inference(leaf)

        # add exploration noise
        if leaf == self.root and self.dirichlet_epsilon > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(priors))
            priors = (1 - self.dirichlet_epsilon
                      ) * priors + self.dirichlet_epsilon * noise

        # Zero out invalid moves
        priors *= leaf.valid_mask
        sum_priors = priors.sum()
        if sum_priors > 1e-12:
            priors /= sum_priors  # normalize

        # Expand each valid move
        for action in leaf.valid_actions:
            # Construct next state
            action = int(action)

            # Child’s prior from policy distribution
            leaf.expand(action, priors[action])

        # Backprop the leaf_value from the policy to this leaf
        leaf.backpropagate(leaf_value)

    def _select_child(self, node: Node) -> Node:
        """
        PUCT selection among the children of 'node'.
        """
        best_a = max(
            (int(a) for a in node.valid_actions if int(a) in node.children),
            key=lambda a: node._get_ucb_score(a, node.children[a].prior),
        )
        return node.children[best_a]

    def _simulate(self, root_node: Node):
        node = root_node
        path: List[Node] = []

        # 1) Traverse down
        while True:
            node.add_virtual_loss()
            path.append(node)

            if node.is_terminal:
                # Terminal leaf => backprop the known terminal value
                node.backpropagate(node.terminal_value)
                break

            if node.is_leaf():
                # Leaf => Expand it, then backprop
                self._expand_and_evaluate(node)
                break

            # Otherwise, pick a child to go deeper
            node = self._select_child(node)

        for n in path:
            n.revert_virtual_loss()
