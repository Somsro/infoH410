import numpy as np
from board2048_ext import Board
from time import time
# from your_env_lib import ParallelEnv   # décommentez selon votre implémentation de ParallelEnv

class Environment:
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.board = Board()
        self.agent = "player"
        self.possible_agents = [self.agent]
        self.start_time = time()

    def clone(self):
        new_env = Environment()
        new_env.start_time = self.start_time
        new_env.agent = self.agent
        new_env.possible_agents = self.possible_agents
        new_env.board = self.board.clone()
        return new_env
    
    def set_board(self, board):
        self.board = board

    def is_done(self):
        return len(self.get_valid_actions()) == 0

    def reset(self, seed=None, options=None):
        self.board = Board()
        observation = np.array(self.board.to_list(), dtype=np.int64)
        self.done = False
        self.truncated = False
        self.score = 0
        self.start_time = time()
        if options :
            self.render("New game! Make your first move.")
        return observation
    
    # Utility functions
    def get_score(self, type="default"):
        if type == "expectimax":
            empty_cells  = self.board.get_emptyCount()          # [0, 16]
            monotonicity = self.board.get_monotonicity()        # [0, 12]
            merge_potential = self.board.get_merge_potential()  # [0, 12]
            smoothness   = self.board.get_smoothness()          # [-72, 0]
            max_log_tile = self.board.get_max_logTile()         # [1, 17]
            raw_score = self.get_score("raw")                   # [0, 2^17]

            # Scale corner bonus with tile value instead of binary
            corner_bonus = max_log_tile ** 3 if self.board.is_max_corner() else 0

            # Penalize merges away from corner: if max tile NOT in corner, apply penalty
            corner_penalty = -(max_log_tile ** 3) * 2 if not self.board.is_max_corner() else 0

            score = (
                200.0 * empty_cells +      # 3200 max
                50.0  * monotonicity +     # 600 max
                30.0  * merge_potential +  # 360 max
                10.0  * smoothness +       # -720 max
                corner_bonus +             # 4913 max
                corner_penalty +           # -9826 max
                0.1 * raw_score            # Small weight to encourage without dominating the heuristic
            )
            return score
        else:
            return sum((2**v if v > 0 else 0) for v in self.board.to_list())
    
    def get_valid_actions(self):
        return [d for d in range(4) if self.board.is_move_valid(d)]

    def get_empty_count(self):
        return int(self.board.get_emptyCount())
    
    def get_empty_cells(self):
        return self.board.get_empty_cells()

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
            return self
            # raise NotImplementedError("Expectimax state representation not implemented yet.")
        
        elif agent_type == "td":
            return self
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def step(self, action: int):
        # prev_score = self.get_score()
        # prev_empty_count = self.get_empty_count()
        # prev_max_log_tile = self.get_max_log_tile()

        merge_value = self.board.sweep(int(action))
        terminated = False
        if merge_value >= 0:
            self.board.place_tile()
            self.get_valid_actions()
            terminated = self.get_valid_actions() == []

        # if moved:
        #     placed = bool(self.board.place_tile()) # place_tile() returns False if it cannot place a new tile, which means the game is over
        #     curr_score = self.get_score()
        #     reward = self.calculate_reward(prev_empty_count, self.get_empty_count(), prev_max_log_tile, self.get_max_log_tile(), prev_valid_moves_count, len(self.get_valid_actions()), self.is_max_corner())
        #     if not placed:
        #         reward = -50.0
        #         terminated = True
        # else:
        #     reward = -100.0  # Heavy penalty for invalid moves (for qlearners it souhldn't happen since we only give valid moves)
        #     curr_score = prev_score

        self.done = terminated
        # self.score = curr_score

        # return observation, reward, terminated, curr_score
        return merge_value, terminated
    
    def simple_step(self, action: int):
        return self.board.sweep(int(action))
    
    def place_tile(self, cell, log_value):
        self.board.force_place_tile(cell, log_value)

    def get_duration(self):
        return time() - self.start_time

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