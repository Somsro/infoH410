from pathlib import Path

# Display settings
DIRECTION_NAMES = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

# Training hyperparameters
NUM_EPISODES = 50000

# Hyperparameters for TD agent
TD_WEIGHTS_PATH = Path("td_weights")
TD_LEARNING_RATE = 0.1
TD_EPS_MIN = 0.01
TD_EPS_DECAY = 0.9999

# Expectimax hyperparameters
EXPECTIMAX_DEPTH = 5
INFINITY = int(1e9)
PROB_CUTOFF = 1e-4

# Testing hyperparameters
TEST_EPISODES = 100