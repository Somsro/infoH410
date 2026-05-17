import numpy as np
import gymnasium as gym
from gymnasium import spaces
from board2048_ext import Board
from time import time
from DQNutils import process_state

class Env2048(gym.Env):
    """Gymanisum environment for the 2048 game."""
    
    def __init__(self):
        super().__init__()
        
        # 4 actions : 0 (Up), 1 (Right), 2 (Down), 3 (Left)
        self.action_space = spaces.Discrete(4)
        
        # Observations space 
        self.observation_space = spaces.Box(
            low=0, 
            high=16, 
            shape=(4, 4), 
            dtype=np.int32
        )
        
        self.board = Board()

        self.start_time = time()
        self.step_count = 0

    def get_score(self):
        return sum((2**v if v > 0 else 0) for v in self.board.to_list())

    def _get_obs(self):
        """Takes the current board state and converts it into a 4x4 NumPy array of integers."""
        board_list = self.board.to_list()
        return np.array(board_list, dtype=np.int32).reshape((4, 4))
    
    def get_state(self, agent):
        """Returns the current state of the environment as a 4x4 NumPy array."""
        return process_state(self._get_obs(), agent.device)

    # Create a new board at the start of each episode
    def reset(self, seed=None, options=None):
        """Resets the game at the beginning of an episode."""
        super().reset(seed=seed) # Required by Gymnasium
        
        self.board = Board()
        
        obs = self._get_obs()
        info = {}

        self.step_count = 0
        self.start_time = time()
        
        return obs, info
    
    def get_valid_actions(self):
        return [i for i in range(4) if self.unwrapped.board.is_move_valid(i)]

    # Step function: apply action, calculate reward, check for termination
    def step(self, action):
        action = int(action)

        # Give the value of the merged tile as reward if > 0, if merge_reward is negative it means the move was invalid (no tiles were merged and the board state didn't change), we give a penalty of -4 in this case to encourage the agent to avoid invalid moves.
        merge_reward = self.board.sweep(action)
        
        terminated = False
        truncated = False
                    
        if merge_reward < 0 :
            # Penalty for invalid move (it shouldn't happen as there is a mask for invalid actions, but we add it just in case to encourage the agent to avoid invalid moves)
            reward = -4
            
        else :
            
            game_continues = self.board.place_tile()
            
            if not game_continues:
                terminated = True
                # Penalty for losing the game (to encourage the agent to avoid losing)
                reward = -1.0 
            else:
                
                # Reward when merging tiles (log of the value of the merged tile, ex: +1 for merging two 2's, +2 for merging two 4's, etc.)
                reward_base = merge_reward if merge_reward > 0 else -0.2
                
                # Small reward for empty tiles (to encourage the agent to keep the board less crowded)
                empty_bonus = self.board.get_emptyCount() * 0.1
                
                # Monotonicity bonus (to encourage the agent to keep the board in a more ordered state, which is often beneficial in 2048)
                mono_bonus = self.board.get_monotonicity() * 0.05
                
                # Penalty if the max tile is not in the corner (to encourage the agent to keep the highest tile in a corner, which is a common strategy in 2048)
                corner_penalty = 0 if self.board.is_max_corner() else -0.5
                
                # Sum all the components to get the final reward for this step
                reward = reward_base + empty_bonus + mono_bonus + corner_penalty

        self.step_count += 1
            
        return self._get_obs(), reward, terminated, truncated

    # We add some helper functions to get the duration of the episode and the number of steps taken, which can be useful for tracking the agent's performance during training.
    def get_duration(self):
        return time() - self.start_time
    
    def get_step_count(self):
        return self.step_count

    # Finally, we add a render function to visualize the current state of the board. This function prints the board in a human-readable format, showing the value of each tile (or a dot for empty tiles). 
    # It also displays a message if provided and indicates that the environment is waiting for the next agent move.
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