from pathlib import Path

# Display settings
DIRECTION_NAMES = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

# Training hyperparameters
NUM_EPISODES = 50000
MAX_TRAINING_TIME_SECONDS = 36000

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

# DQN hyperparameters
DQN_MODEL_PATH = Path("dqn_model.pth")
DQN_BATCH_SIZE = 256
DQN_GAMMA = 0.99
DQN_EPS_START = 0.9
DQN_EPS_END = 0.01
DQN_EPS_DECAY = 1000000
DQN_TAU = 0.005
DQN_LR = 1e-4