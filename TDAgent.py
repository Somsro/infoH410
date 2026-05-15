import numpy as np
from pathlib import Path

def _slide_row_left(row):
    """
    Merge and slide a row (log2-encoded values) to the left.
    Returns (merged_row, merge_score) where merge_score = sum of new merged tile values.
    E.g. two tiles of log2-value 3 (=8) merge into log2-value 4 (=16), contributing 16 to the score.
    """
    tiles = [t for t in row if t != 0]
    result, score = [], 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] + 1          # log2(2x) = log2(x) + 1
            result.append(merged)
            score += 2 ** merged           # actual tile value earned
            i += 2
        else:
            result.append(tiles[i])
            i += 1
    result += [0] * (4 - len(result))
    return result, score

def compute_after_state(board_list, action):
    """
    Simulate a sweep without placing a new tile (after-state computation).

    board_list : flat list of 16 log2-encoded tile values (0 = empty).
    action     : 0=up, 1=right, 2=down, 3=left
    Returns    : (after_board_list, moved, merge_score)
                 merge_score is the raw 2048 game score earned by this sweep
                 (sum of all newly merged tile values).
    """
    b = [list(board_list[r * 4:(r + 1) * 4]) for r in range(4)]
    total_score = 0

    if action == 3:  # Left
        nb = []
        for r in range(4):
            row, s = _slide_row_left(b[r])
            nb.append(row)
            total_score += s
    elif action == 1:  # Right: reverse row, slide left, reverse back
        nb = []
        for r in range(4):
            row, s = _slide_row_left(b[r][::-1])
            nb.append(row[::-1])
            total_score += s
    elif action == 0:  # Up: transpose, slide left, transpose back
        cols = [[b[r][c] for r in range(4)] for c in range(4)]
        sc = []
        for c in range(4):
            col, s = _slide_row_left(cols[c])
            sc.append(col)
            total_score += s
        nb = [[sc[c][r] for c in range(4)] for r in range(4)]
    else:             # Down: transpose, slide right, transpose back
        cols = [[b[r][c] for r in range(4)] for c in range(4)]
        sc = []
        for c in range(4):
            col, s = _slide_row_left(cols[c][::-1])
            sc.append(col[::-1])
            total_score += s
        nb = [[sc[c][r] for c in range(4)] for r in range(4)]

    flat = [nb[r][c] for r in range(4) for c in range(4)]
    return flat, flat != list(board_list), total_score


class NTupleNetwork:
    """
    N-tuple network value function approximator for 2048.

    Uses 6 base 6-tuple patterns, each expanded via 8-way board symmetry
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
        (0, 1, 2, 3, 4, 8),
        (0, 1, 4, 5, 8, 9),
        (1, 2, 5, 6, 9, 10),
    ]

    def __init__(self, dynamic_lr=True, learning_rate=0.0025):
        self.patterns    = self._generate_symmetric_patterns()
        self.dynamic_lr  = dynamic_lr
        self.alpha       = learning_rate
        lut_size         = self.NUM_VALUES ** 6

        self.luts = [np.zeros(lut_size, dtype=np.float32) for _ in self.patterns]

        if self.dynamic_lr:
            self.counts     = [np.zeros(lut_size, dtype=np.uint32) for _ in self.patterns]
            self.MIN_VISITS = 50   # trust early estimates less — avoids wild updates

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
            idx = idx * self.NUM_VALUES + board[cell]
        return idx

    def evaluate(self, board):
        """Estimate the value of a board state (sum over all lookup tables)."""
        return sum(lut[self._index(board, p)]
                   for lut, p in zip(self.luts, self.patterns))

    def update_weights(self, board, td_error):
        """
        Update each active lookup table entry using td_error directly.

        Dynamic LR : lr = 1 / max(count, MIN_VISITS) per entry.
        Fixed LR   : lr = alpha / n_base_patterns.
        """
        n = len(self.BASE_PATTERNS)
        for i, (lut, p) in enumerate(zip(self.luts, self.patterns)):
            idx = self._index(board, p)
            if self.dynamic_lr:
                self.counts[i][idx] += 1
                lr = 1.0 / max(self.counts[i][idx], self.MIN_VISITS)
            else:
                lr = self.alpha / n
            lut[idx] += lr * td_error

    # Persistence

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
                 epsilon_min=0.001, epsilon_decay=0.99995,
                 dynamic_lr=True):
        self.network       = NTupleNetwork(dynamic_lr=dynamic_lr,
                                           learning_rate=learning_rate)
        self.alpha         = learning_rate
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.episode_final_scores = []

    def select_action(self, board_list, valid_actions):
        """
        Pick an action using epsilon-greedy after-state evaluation.
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)

        best_action = valid_actions[0]
        best_value  = -float('inf')
        for action in valid_actions:
            after_board, moved, _ = compute_after_state(board_list, action)
            if moved:
                value = self.network.evaluate(after_board)
                if value > best_value:
                    best_value  = value
                    best_action = action
        return best_action

    def update(self, prev_after_board, reward, curr_after_board, done):
        """
        Perform a TD(0) weight update.

        prev_after_board : after-state from the previous time step.
        reward           : raw merge score earned by that action.
        curr_after_board : after-state for the current action (None if terminal).
        done             : True when the episode has ended.
        Returns          : TD error (scalar).
        """
        if prev_after_board is None:
            return 0.0

        v_prev = self.network.evaluate(prev_after_board)

        # Raw reward
        # relative to v_next which can be in the thousands
        if done:
            td_error = float(reward) - v_prev   # v_next = 0 at terminal
        else:
            v_next   = self.network.evaluate(curr_after_board)
            td_error = float(reward) + v_next - v_prev

        self.network.update_weights(prev_after_board, td_error)
        return td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Persistence

    def save(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.network.save(filepath)

    def load(self, filepath):
        self.network.load(filepath)
