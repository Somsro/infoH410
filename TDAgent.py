import numpy as np
from pathlib import Path


# ── Board simulation ──────────────────────────────────────────────────

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


# ── N-Tuple Network ───────────────────────────────────────────────────

class NTupleNetwork:
    """
    N-tuple network value function approximator for 2048.

    Uses 4 base 4-tuple patterns, each expanded via 8-way board symmetry
    (4 rotations × horizontal reflection), yielding up to 32 unique lookup
    tables.  Each cell holds a log2-encoded tile value in [0, NUM_VALUES).

    Board cell indices (row-major):
         0  1  2  3
         4  5  6  7
         8  9 10 11
        12 13 14 15

    Reference: Szubert & Jaskowski (2014), "Temporal Difference Learning of
    N-Tuple Networks for the Game 2048".
    """

    NUM_VALUES = 16  # log2 tile values 0..15 (0 = empty)

    # Four base 4-tuples covering different spatial relationships.
    BASE_PATTERNS = [
        (0, 1, 2, 3),   # horizontal row
        (0, 1, 4, 5),   # 2×2 square
        (0, 1, 5, 6),   # 2×2 diagonal shift
        (0, 4, 5, 9),   # zigzag
    ]

    def __init__(self):
        self.patterns = self._generate_symmetric_patterns()
        lut_size = self.NUM_VALUES ** len(self.BASE_PATTERNS[0])
        self.luts = [np.zeros(lut_size, dtype=np.float64) for _ in self.patterns]

    # ── Symmetry helpers ──────────────────────────────────────────────

    @staticmethod
    def _rotate_idx(i):
        """Rotate a cell index 90° clockwise on the 4×4 grid."""
        row, col = i // 4, i % 4
        return col * 4 + (3 - row)

    @staticmethod
    def _reflect_idx(i):
        """Reflect a cell index horizontally on the 4×4 grid."""
        row, col = i // 4, i % 4
        return row * 4 + (3 - col)

    def _apply(self, pattern, fn):
        return tuple(fn(i) for i in pattern)

    def _generate_symmetric_patterns(self):
        """Expand each base pattern to all unique 8-way symmetry variants."""
        all_patterns, seen = [], set()
        for base in self.BASE_PATTERNS:
            p = base
            for _ in range(4):                                   # 4 rotations
                for variant in (p, self._apply(p, self._reflect_idx)):
                    if variant not in seen:
                        seen.add(variant)
                        all_patterns.append(variant)
                p = self._apply(p, self._rotate_idx)
        return all_patterns

    # ── Core operations ───────────────────────────────────────────────

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

    def update_weights(self, board, delta):
        """Add delta to the active entry of every lookup table for this board."""
        for lut, p in zip(self.luts, self.patterns):
            lut[self._index(board, p)] += delta

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, filepath):
        np.save(filepath, np.array(self.luts, dtype=object), allow_pickle=True)

    def load(self, filepath):
        self.luts = list(np.load(filepath, allow_pickle=True))


# ── TD Agent ─────────────────────────────────────────────────────────

class TDAgent:
    """
    TD(0) agent with n-tuple network for 2048.

    Learns a value function over board after-states (the board after a sweep
    but before the random tile is placed).  The update rule is:

        V(s_after) += (α / N) * [r + V(s'_after) - V(s_after)]

    where N is the number of lookup tables, so the effective per-state update
    magnitude stays α * td_error regardless of how many tuples are used.

    Action selection is ε-greedy over after-state values:
    the action whose resulting after-state has the highest estimated value is
    chosen with probability (1 - ε), and a random valid action otherwise.
    """

    def __init__(self, learning_rate=0.01, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.9995):
        self.network       = NTupleNetwork()
        self.alpha         = learning_rate
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.episode_final_scores = []

    # ── Action selection ──────────────────────────────────────────────

    def select_action(self, board_list, valid_actions):
        """
        Pick an action using ε-greedy after-state evaluation.

        board_list    : flat list of 16 log2-encoded tile values.
        valid_actions : list of valid action indices (non-empty).
        Returns       : chosen action index.
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

    # ── Learning ──────────────────────────────────────────────────────

    def update(self, prev_after_board, reward, curr_after_board, done):
        """
        Perform a TD(0) weight update.

        prev_after_board : after-state from the previous time step.
        reward           : raw merge score earned by the action that produced prev_after_board.
        curr_after_board : after-state for the current action (None if terminal).
        done             : True when the episode has ended.
        Returns          : TD error (scalar, useful for monitoring).
        """
        v_prev = self.network.evaluate(prev_after_board)
        v_next = 0.0 if done else self.network.evaluate(curr_after_board)

        td_error = reward + v_next - v_prev
        delta    = self.alpha * td_error / len(self.network.patterns)
        self.network.update_weights(prev_after_board, delta)
        return td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.network.save(filepath)

    def load(self, filepath):
        self.network.load(filepath)
