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
        if options :
            self.render("New game! Make your first move.")
        return observation
    
    # Utility functions
    def get_score(self, board):
        return sum((2**v if v > 0 else 0) for v in board.to_list())
    
    def get_valid_actions(self):
        return [d for d in range(4) if self.board.is_move_valid(d)]

    def get_empty_count(self):
        return int(self.board.get_emptyCount())

    def get_max_log_tile(self):
        return int(self.board.get_max_logTile())

    def is_max_corner(self):
        return bool(self.board.is_max_corner())
    
    def calculate_reward(self, prev_empty_count, curr_empty_count, prev_max_log_tile, curr_max_log_tile, prev_valid_moves_count, curr_valid_moves_count, is_max_corner):
        reward = 0.0

        # Reward for creating empty spaces and higher tiles:
        reward += (curr_empty_count - prev_empty_count) * 5.0  # Reward 5 for each empty space created
        reward += (curr_max_log_tile - prev_max_log_tile) * 5.0  # Reward 2*(difference in log tile values) for creating higher tiles

        # Reward for increasing the number of valid moves:
        if curr_valid_moves_count > prev_valid_moves_count:
            reward += 1.0
        elif curr_valid_moves_count < prev_valid_moves_count:
            reward -= 2.0

        # Reward for keeping the maximum tile in a corner:
        if is_max_corner:
            reward += 5.0

        return reward
    
    # Logic functions
    def get_state(self, agent_type):
        if agent_type == "qlearner":
            empty_count = self.get_empty_count()
            max_log_tile = self.get_max_log_tile()
            if (max_log_tile > 13): # if the max tile is greater than 8192, we consider it as 8192 since we want to limit the state space and it doesn't matter for the learning process (the agent just needs to know that it has reached a very high tile)
                raise NotImplementedError("State representation for qlearner does not support max_log_tile > 13 (tile 8192)")
            is_max_corner = self.is_max_corner()
            valid_moves_count = len(self.get_valid_actions())

            state = empty_count
            state = state * 14 + max_log_tile # 14 bc max_log_tile can be from 0 to 13 (for tile 8192)
            state = state * 5 + valid_moves_count
            state = state * 2 + int(is_max_corner)

            return state


        elif agent_type == "expectimax":
            raise NotImplementedError("Expectimax state representation not implemented yet.")
        
        elif agent_type == "deeplearning":
            raise NotImplementedError("Deep learning state representation not implemented yet.")
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def step(self, action: int, prev_valid_moves_count: int):
        prev_score = self.get_score(self.board)
        prev_empty_count = self.get_empty_count()
        prev_max_log_tile = self.get_max_log_tile()

        moved = bool(self.board.sweep(int(action))) #sweep() returns False if the move was invalid (no tiles moved or merged)
        reward = 0.0
        terminated = False

        if moved:
            placed = bool(self.board.place_tile()) # place_tile() returns False if it cannot place a new tile, which means the game is over
            curr_score = self.get_score(self.board)
            reward = self.calculate_reward(prev_empty_count, self.get_empty_count(), prev_max_log_tile, self.get_max_log_tile(), prev_valid_moves_count, len(self.get_valid_actions()), self.is_max_corner())
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

    def render(self, message: str = "", over: bool = False) -> None:
        representation = [2**self.board.to_list()[i] if self.board.to_list()[i] > 0 else "   ." for i in range(16)]
        for i in range(4):
            row = representation[i*4:(i+1)*4]
            print(' '.join(f"{v:4}" for v in row))
        if message:
            print(message)
        if over:
            return
        print("Waiting for next agent move...  •  Ctrl-C to quit")

        def close(self):
            pass