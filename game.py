import sys
from environment import Environment
from TDAgent import TDAgent, train_td
from expectimax import ExpectimaxAgent
from tracking import plot_tracking
from PARAMETERS import *

def main(agent_type) -> None:

    if agent_type == "expectimax":
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
            plot_tracking(agent.episode_step_counts, agent.episode_final_scores, windows_count=20, filename="td_learning_tracking_data.png")

    elif agent_type == "dql":
        print("DQL agent not implemented yet.") # TODO
        sys.exit(1)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Play one game with the trained agent and render it
    env = Environment()
    done = False
    env.reset()
    
    while not done: # done is set to True in env.step() when the game is over

        # List of useful getters 
        state = env.get_state(agent_type) # get the current state for the agent

        # Agents take actions based on the state and valid moves
        action = agent.select_action(state) 

        # Step in the environment
        _, done = env.step(action)

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