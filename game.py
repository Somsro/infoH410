import sys
from pathlib import Path
from environment import Environment
from qleaners import Qlearner
import matplotlib.pyplot as plt

#Training hyperparameters
NUM_EPISODES = 50000


# Hyperparameters for Q-learner
GAMMA = 0.99
EPS_MIN = 0.01
EPS_DECAY = 0.9995
STATE_SIZE = 2240 #16 (empty_count) * 14 (max_log_tile) * 5 (valid_moves_count) * 2 (is_max_corner)
QTABLE_PATH = Path("qtable.npy")

# ── Platform-specific single-keypress reading ────────────────────────
if sys.platform == "win32":
    import msvcrt

    def get_arrow() -> int | None:
        """Returns 0=up 1=right 2=down 3=left or None for other keys."""
        ch = msvcrt.getch()
        if ch == b'\xe0':
            ch = msvcrt.getch()
            return {b'H': 0, b'M': 1, b'P': 2, b'K': 3}.get(ch)
        if ch == b'\x03':
            raise KeyboardInterrupt
        return None
else:
    import tty
    import termios

    def get_arrow() -> int | None:
        """Returns 0=up 1=right 2=down 3=left or None for other keys."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == '\x03':
                raise KeyboardInterrupt
            if ch == '\x1b':
                ch2 = sys.stdin.read(1)
                ch3 = sys.stdin.read(1)
                if ch2 == '[':
                    return {'A': 0, 'C': 1, 'B': 2, 'D': 3}.get(ch3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return None


# ── Display ──────────────────────────────────────────────────────────

DIRECTION_NAMES = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

# ── Utility functions ────────────────────────────────────────────────



# ── Game loop ────────────────────────────────────────────────────────

def train_qlearner(nb_episodes) -> Qlearner:
    agent = Qlearner(
            action_size=4,
            state_size=STATE_SIZE,
            learning_rate=0.1,  # dynamic learning rate
            gamma=GAMMA, # How much the agent values future rewards over immediate rewards. A gamma of 0 will make the agent only care about immediate rewards, while a gamma close to 1 will make it strive for long-term high rewards.
            epsilon=1.0,
            epsilon_min=EPS_MIN,
            epsilon_decay=EPS_DECAY,
        )
    
    env = Environment()

    for i in range(nb_episodes): # train for nb_episodes episodes
        print(f"Game {i+1} starting...")
        done = False
        env.reset(options=False)

        while not done: #done is set to True in env.step() when the game is over

            state = env.get_state("qlearner") # get the current state for the agent

            valid_moves = env.get_valid_actions() # get the valid moves for the current state (will always return at least 1 since done would be True otherwise)

            action = agent.select_action(state, valid_moves) # Agents take actions based on the state and valid moves

            # Step in the environment
            obs, reward, done, score = env.step(action, len(env.get_valid_actions())) #reward = score difference after taking the action.

            new_state = env.get_state("qlearner") # get the new state after taking the action
            next_valid_actions = env.get_valid_actions() # get the valid moves for the new state (will always return at least 1 since done would be True otherwise)

            # Agent learns
            agent.update(state, action, new_state, reward, score, done, next_valid_actions)

            # Render the new state of the game
            if (not done) :
                #env.render(f"Action: {DIRECTION_NAMES[action]}, Actual Score: {score}")
                pass
            else:
                #env.render(f"Game Over! Final Score: {score}")
                pass

        print(f"Game {i+1} ended with score: {score}")
        #agent.print_scores(i) # print the score of the episode and the epsilon value to see the progress of the agent

    agent.save_qtable(QTABLE_PATH)
    print(f"Saved qtable to {QTABLE_PATH}")
    return agent
    




def main(agent_type) -> None:

    if agent_type == "qlearner":
        if QTABLE_PATH.exists():
            agent = Qlearner(
                action_size=4,
                state_size=STATE_SIZE,
                learning_rate=0.0,
                gamma=GAMMA,
                epsilon=0.0,  # no exploration since we are loading a trained qtable
                epsilon_min=EPS_MIN,
                epsilon_decay=EPS_DECAY,
            )
            agent.load_qtable(QTABLE_PATH)
            print(f"Loaded qtable from {QTABLE_PATH}")
        else:
            agent = train_qlearner(NUM_EPISODES)
            plt.plot(agent.episode_final_scores)
            plt.xlabel("Episode")
            plt.ylabel("Score")
            plt.title("Q-learner 2048 Training")
            plt.show()

    elif agent_type == "expectimax":
        #agent = ...
        pass  # TODO: implement
    elif agent_type == "td":
        #agent = ...
        pass  # TODO: implement
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Play one game with the trained agent and render it
    env = Environment()
    done = False
    env.reset()
    
    while not done: #done is set to True in env.step() when the game is over

        # List of useful getters 
        state = env.get_state(agent_type) # get the current state for the agent
        valid_moves = env.get_valid_actions() # get the valid moves for the current state (will always return at least 1 since done would be True otherwise)


        # Agents take actions based on the state and valid moves
        action = agent.select_action(state, valid_moves) 
        # Step in the environment
        obs, reward, done, score = env.step(action, len(env.get_valid_actions())) #reward = score difference after taking the action.

        # List of useful getters
        new_state = env.get_state(agent_type) # get the new state after taking the action
        next_valid_actions = env.get_valid_actions() # get the valid moves for the new state (will always return at least 1 since done would be True otherwise)

        # Render the new state of the game
        if (not done) :
            env.render(f"Action: {DIRECTION_NAMES[action]}, Actual Score: {score}")
        else:
            env.render(f"Game Over! Final Score: {score}")


# Call main with agent_type specified in command line argument
if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            agent_type = sys.argv[1]
            main(agent_type)
        else:
            print("Please specify an agent type.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nBye!")