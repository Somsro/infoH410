import numpy as np
from board2048_ext import Board
# from your_env_lib import ParallelEnv   # décommentez selon votre implémentation de ParallelEnv

class Environment:
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.board = Board()
        self.agent = "player"
        self.possible_agents = [self.agent]

    def reset(self, seed=None, options=None):
        self.board = Board()
        observation = np.array(self.board.to_list(), dtype=np.int64)
        self.reward = 0
        self.done = False
        self.truncated = False
        self.score = 0
        self.render("New game! Make your first move.")
        return observation
    
    def get_state(self, agent_type):
        if agent_type == "qlearner":
            raise NotImplementedError("Qlearner state representation not implemented yet.")
        elif agent_type == "expectimax":
            raise NotImplementedError("Expectimax state representation not implemented yet.")
        elif agent_type == "deeplearning":
            raise NotImplementedError("Deep learning state representation not implemented yet.")
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def get_score(self, board):
        return sum((2**v if v > 0 else 0) for v in board.to_list())
    
    def get_valid_actions(self):
        return [d for d in range(4) if self.board.is_move_valid(d)]

    def step(self, action: int):
        prev_score = self.get_score(self.board)

        moved = bool(self.board.sweep(int(action))) #sweep() returns False if the move was invalid (no tiles moved or merged)
        reward = 0.0
        terminated = False

        if moved:
            placed = bool(self.board.place_tile()) # place_tile() returns False if it cannot place a new tile, which means the game is over
            curr_score = self.get_score(self.board)
            reward = float(curr_score - prev_score)
            if not placed:
                reward = -50.0
                terminated = True
        else:
            reward = -100.0  # Heavy penalty for invalid moves (for qlearners it souhldn't happen since we only give valid moves)
            curr_score = prev_score

        observation = np.array(self.board.to_list(), dtype=np.int64)

        self.reward = reward
        self.done = terminated
        self.score = curr_score

        return observation, reward, terminated, curr_score

    def render(self, message: str = "") -> None:
        representation = [2**self.board.to_list()[i] if self.board.to_list()[i] > 0 else "   ." for i in range(16)]
        for i in range(4):
            row = representation[i*4:(i+1)*4]
            print(' '.join(f"{v:4}" for v in row))
        if message:
            print(message)
        print("Arrow keys to move  •  Ctrl-C to quit")

        def close(self):
            pass