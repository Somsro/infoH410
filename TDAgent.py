import numpy as np
from pathlib import Path
from environment import Environment
from PARAMETERS import TD_LEARNING_RATE, TD_EPS_MIN, TD_EPS_DECAY, TD_WEIGHTS_PATH
from tracking import save_tracking

class NTupleNetwork:
    """
    N-tuple network value function approximator for 2048.

    Uses 4 base 6-tuple patterns, each expanded via 8-way board symmetry
    (4 rotations x horizontal reflection), yielding up to 48 unique lookup
    tables. Each cell holds a log2-encoded tile value in [0, NUM_VALUES].

    With dynamic LR enabled, each lookup table entry tracks its own visit
    count and uses lr = 1/count instead of a fixed alpha, guaranteeing
    convergence per the tabular TD learning theory.

    Board cell indices (row-major):
         0  1  2  3
         4  5  6  7
         8  9 10 11
        12 13 14 15

    Reference: Szubert & Jaskowski (2014), "Temporal Difference Learning of
    N-Tuple Networks for the Game 2048".
    """

    NUM_VALUES = 16

    BASE_PATTERNS = [
        (0, 1, 2, 3, 4, 5),
        (4, 5, 6, 7, 8, 9),
        (0, 1, 2, 4, 5, 6),
        (4, 5, 6, 8, 9, 10),
    ]


    def __init__(self, dynamic_lr=True, learning_rate=0.0025):
        self.patterns    = self._generate_symmetric_patterns()
        self.dynamic_lr  = dynamic_lr
        self.alpha       = learning_rate
        self.alpha_floor = 0.0125
        lut_size         = self.NUM_VALUES ** 6

        self.luts = [np.zeros(lut_size, dtype=np.float32) for _ in self.patterns]

        if self.dynamic_lr:
            # Track how often each LUT entry is visited so we can reduce learning rate over time.
            self.counts     = [np.zeros(lut_size, dtype=np.uint32) for _ in self.patterns]
            # Do not let the per-entry learning rate decay too fast on the first few updates. (avoid that first visit having a huge impact on the weights)
            self.MIN_VISITS = 100

    # Symmetry helpers

    @staticmethod
    def _rotate_idx(i):
        """Rotate a cell index 90° clockwise on the 4x4 grid."""
        row, col = i // 4, i % 4
        return col * 4 + (3 - row)

    @staticmethod
    def _reflect_idx(i):
        """Reflect a cell index horizontally on the 4x4 grid."""
        row, col = i // 4, i % 4
        return row * 4 + (3 - col)

    def _apply(self, pattern, fn):
        return tuple(fn(i) for i in pattern)

    def _generate_symmetric_patterns(self):
        """Expand each base pattern to all unique 8-way symmetry variants."""
        all_patterns, seen = [], set()
        for base in self.BASE_PATTERNS:
            p = base
            for _ in range(4):
                for variant in (p, self._apply(p, self._reflect_idx)):
                    if variant not in seen:
                        seen.add(variant)
                        all_patterns.append(variant)
                p = self._apply(p, self._rotate_idx)
        return all_patterns

    # Core operations

    def _index(self, board, pattern):
        """Convert the cells selected by a pattern into a lookup table index."""
        idx = 0
        for cell in pattern:
            # Ensure tile log-value never exceeds 15 (for 16^6 LUT)
            val = min(board[cell], self.NUM_VALUES - 1)
            idx = idx * self.NUM_VALUES + val
        return idx

    def evaluate(self, board):
        """Estimate the value of a board state (sum over all lookup tables)."""
        return sum(lut[self._index(board, p)]
                   for lut, p in zip(self.luts, self.patterns))

    def update_weights(self, board, td_error):
        n_iso = len(self.patterns) // len(self.BASE_PATTERNS)
        total = 0.0
        for i, (lut, p) in enumerate(zip(self.luts, self.patterns)):
            idx = self._index(board, p)
            if self.dynamic_lr: #Update learning rate in case of dynamic lr
                self.counts[i][idx] += 1
                lr = max(1.0 / max(self.counts[i][idx], self.MIN_VISITS),
                        self.alpha_floor)
            else:
                lr = self.alpha / n_iso
            lut[idx] += lr * td_error
            total += float(lut[idx])
        return total

    # Analysis utility

    def save(self, filepath):
        filepath = Path(filepath).with_suffix('')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Compress all LUTs into a single .npz file for storage efficiency
        np.savez_compressed(filepath, luts=np.stack(self.luts))

    def load(self, filepath):
        # np.load needs the .npz extension to find the file
        filepath = Path(filepath).with_suffix('.npz')
        data      = np.load(filepath)
        stacked   = data['luts']
        self.luts = [stacked[i] for i in range(len(stacked))]
        if self.dynamic_lr:
            lut_size    = self.NUM_VALUES ** 6
            self.counts = [np.zeros(lut_size, dtype=np.uint32) for _ in self.patterns]


class TDAgent:
    """
    TD(0) agent with n-tuple network for 2048.

    Learns a value function over board after-states (the board after a sweep
    but before the random tile is placed). The update rule is:

        V(s_after) += lr * [r + V(s'_after) - V(s_after)]

    Action selection is epsilon-greedy over after-state values.
    """

    def __init__(self, learning_rate=0.0025, epsilon=1.0,
                 epsilon_min=0.001, epsilon_decay=0.9995,
                 dynamic_lr=True):
        self.network       = NTupleNetwork(dynamic_lr=dynamic_lr,
                                           learning_rate=learning_rate)
        self.alpha         = learning_rate
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.agent_type = "td"

        self.episode_final_scores = []
        self.episode_durations = []
        self.episode_step_counts = []

    def select_action(self, env):
        """Choose a move with epsilon-greedy exploration over after-state values.
        (after-states are the board configurations immediately after the agent's move, without the random tile added yet)
        return int action index in [0, 3] or None if no valid moves."""

        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None

        if np.random.random() < self.epsilon: # Explore: choose a random valid action
            return np.random.choice(valid_actions)

        best_action = valid_actions[0]
        best_value  = -float('inf')
        for action in valid_actions: # Evaluate the after-state for each valid action to choose the best one
            new_env = env.clone()
            reward  = float(new_env.simple_step(action)) #Make sweep but don't add random tile yet.
            value   = reward + self.network.evaluate(new_env.board.to_list()) #TD agent uses the board.to_list() as state
            if value > best_value:
                best_value  = value
                best_action = action
        return best_action

    def update(self, env, reward, new_env, done):
        """
        Perform a TD(0) weight update.

        env     : after-state from the previous time step.
        reward  : raw merge score earned by that action.
        new_env : after-state for the current action (None if terminal).
        done    : True when the episode has ended.
        Returns : TD error (scalar).
        """
        if new_env is None:
            return 0.0

        v_prev = self.network.evaluate(env.board.to_list())

        # Raw reward
        # relative to v_next which can be in the thousands
        if done:
            td_error = float(reward) - v_prev   # v_next = 0 at terminal
        else:
            v_next   = self.network.evaluate(new_env.board.to_list())
            #Gamma = 1 so future states are as much considered as immediate reward.
            td_error = float(reward) + v_next - v_prev

        self.network.update_weights(env.board.to_list(), td_error)
        return td_error
    
    def learn_from_episode(self, path):
        """
        Backward TD update through the full episode path.
        path: list of (after_state_list, reward, done) in forward order.
        """
        target = 0.0
        #Recursive update : V(s) += alpha * [r + V(s') - V(s)] where s' is the next state in the episode path.
        for board_list, reward, done in reversed(path[:-1]):
            v_current  = self.network.evaluate(board_list)
            td_error   = target - v_current
            target     = float(reward) + self.network.update_weights(board_list, td_error)

    def decay_epsilon(self):
        """Apply exponential decay to the exploration rate if it's above the minimum."""

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Persistence

    def save(self, filepath):
        """Save the learned NTuple network."""

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.network.save(filepath)

    def load(self, filepath):
        """Load the NTuple network."""

        self.network.load(filepath)

def train_td(nb_episodes) -> TDAgent:
    """Train a TD(0) agent for a fixed number of episodes given in arg."""

    agent = TDAgent(
        learning_rate=TD_LEARNING_RATE,
        epsilon=1.0,
        epsilon_min=TD_EPS_MIN,
        epsilon_decay=TD_EPS_DECAY,
        dynamic_lr=True,
    )
    env = Environment()

    for episode in range(nb_episodes):
        env.reset(options=False)
        done = False
        path = []

        while not done:
            action      = agent.select_action(env)
            after_env   = env.clone()
            reward      = after_env.simple_step(action)
            
            board_list  = after_env.board.to_list()
            _, done     = env.step(action)
            path.append((board_list, reward, done))

        # Learn backwards through the episode
        agent.learn_from_episode(path)

        # Clear the path explicitly to help Python's garbage collection
        path.clear() 

        agent.decay_epsilon()
        agent.episode_final_scores.append(env.get_score())
        agent.episode_durations.append(env.get_duration())
        agent.episode_step_counts.append(env.get_step_count())

        if (episode + 1) % 100 == 0:
            recent = agent.episode_final_scores[-100:]
            avg    = sum(recent) / len(recent)
            print(f"Episode {episode+1}/{nb_episodes} | Avg Score: {avg:.0f} | Epsilon: {agent.epsilon:.4f}")

    agent.save(TD_WEIGHTS_PATH)
    save_tracking(agent.episode_step_counts, agent.episode_durations, agent.episode_final_scores, "td_learning_tracking_data.npz")
    return agent