import sys
from pathlib import Path
from environment import Environment
from qleaners import Qlearner
from TDAgent import TDAgent
import matplotlib.pyplot as plt
from expectimax import ExpectimaxAgent
from tracking import plot_tracking, save_tracking

#Training hyperparameters
NUM_EPISODES = 50000

# Hyperparameters for Q-learner
GAMMA = 0.99
EPS_MIN = 0.01
EPS_DECAY = 0.9995
STATE_SIZE = 2240 #16 (empty_count) * 14 (max_log_tile) * 5 (valid_moves_count) * 2 (is_max_corner)
QTABLE_PATH = Path("qtable.npy")

# Hyperparameters for TD agent
TD_WEIGHTS_PATH = Path("td_weights")
TD_LEARNING_RATE = 0.1
TD_EPS_MIN = 0.01
TD_EPS_DECAY = 0.9999

# Expectimax hyperparameters
EXPECTIMAX_DEPTH = 6

# ── Display ──────────────────────────────────────────────────────────

DIRECTION_NAMES = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}


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


def train_td(nb_episodes) -> TDAgent:
    agent = TDAgent(
        learning_rate=TD_LEARNING_RATE,
        epsilon=1.0,
        epsilon_min=TD_EPS_MIN,
        epsilon_decay=TD_EPS_DECAY,
        dynamic_lr=True,
    )
    env = Environment()

    for episode in range(nb_episodes):
        env.reset(options=False)
        done = False
        path = []

        while not done:
            action      = agent.select_action(env)
            after_env   = env.clone()
            reward      = after_env.simple_step(action)
            
            board_list  = after_env.board.to_list()
            _, done     = env.step(action)
            path.append((board_list, reward, done))

        # Learn backwards through the episode
        agent.learn_from_episode(path)

        # Clear the path explicitly to help Python's garbage collection
        path.clear() 

        agent.decay_epsilon()
        agent.episode_final_scores.append(env.get_score())
        agent.episode_durations.append(env.get_duration())

        if (episode + 1) % 100 == 0:
            recent = agent.episode_final_scores[-100:]
            avg    = sum(recent) / len(recent)
            print(f"Episode {episode+1}/{nb_episodes} | Avg Score: {avg:.0f} | Epsilon: {agent.epsilon:.4f}")

    agent.save(TD_WEIGHTS_PATH)
    save_tracking(agent.episode_durations, agent.episode_final_scores, "td_learning_tracking_data.npz")
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
        agent = ExpectimaxAgent(depth=EXPECTIMAX_DEPTH)

    elif agent_type == "td":
        weights_file = TD_WEIGHTS_PATH.with_suffix('.npz')
        if weights_file.exists():
            agent = TDAgent(
                learning_rate=0.0,
                epsilon=0.0,
                epsilon_min=0.0,
                epsilon_decay=1.0,
                dynamic_lr=False,   # no counts needed for inference
            )
            agent.load(TD_WEIGHTS_PATH)
            print(f"Loaded TD weights from {weights_file}")
        else:
            agent = train_td(NUM_EPISODES)
            plot_tracking(agent.episode_durations, agent.episode_final_scores, "td_learning_tracking_data.png")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Play one game with the trained agent and render it
    env = Environment()
    done = False
    env.reset()
    
    while not done: #done is set to True in env.step() when the game is over

        # List of useful getters 
        state = env.get_state(agent_type) # get the current state for the agent

        # Agents take actions based on the state and valid moves
        action = agent.select_action(state) 

        # Step in the environment
        _, done = env.step(action) #reward = score difference after taking the action.

        # Render the new state of the game
        if (not done) :
            env.render(f"Action: {DIRECTION_NAMES[action]}, Actual Score: {env.get_score()}")
        else:
            env.render(f"Game Over! Final Score: {env.get_score()}")


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