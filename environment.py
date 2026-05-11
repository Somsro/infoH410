import numpy as np
from board2048_ext import Board, render
# from your_env_lib import ParallelEnv   # décommentez selon votre implémentation de ParallelEnv

class Environment:
    metadata = {"render_modes": ["human"]}

    def __init__(self, board: Board | None = None):
        self.board = board or Board()
        self.agent = "player"
        self.possible_agents = [self.agent]

    def reset(self, seed=None, options=None):
        self.board = Board()
        observation = np.array(self.board.to_list(), dtype=np.int64)
        self.reward = 0.0
        self.done = False
        self.truncated = False
        self.info = {}
        return observation
    
    def get_state(self, agent_type):
        if agent_type == "qlearners":
            return self.board.get_state_id()  # returns an int representing the current state
        elif agent_type == "expectimax":
            raise NotImplementedError("Expectimax agent not implemented yet.")
        elif agent_type == "deeplearning":
            raise NotImplementedError("Deep learning agent not implemented yet.")
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def step(self, action: int):
        prev_score = self.board.get_score()

        moved = bool(self.board.sweep(int(action)))
        reward = 0.0
        terminated = False

        if moved:
            placed = bool(self.board.place_tile())
            curr_score = self.board.get_score()
            reward = float(curr_score - prev_score)
            if not placed:
                terminated = True

        observation = np.array(self.board.to_list(), dtype=np.int64)
        truncated = False
        info = {}

        self.reward = reward
        self.done = terminated
        self.truncated = truncated
        self.info = info

        return observation, reward, terminated, truncated, info

    def render(self, message: str = ""):
        render(self.board, message)

    def close(self):
        pass