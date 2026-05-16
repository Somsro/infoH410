import sys
from TDAgent import TDAgent
from expectimax import ExpectimaxAgent
from environment import Environment
from tracking import test_statistics, save_test_statistics
from PARAMETERS import TEST_EPISODES, TD_WEIGHTS_PATH, EXPECTIMAX_DEPTH, DQN_MODEL_PATH
from DQNenv import Env2048
from DQNAgent import DQNAgent

def test_loop(agent, agent_type, env) -> None:
    header = f"{'Game':<10} | {'Score':<12} | {'Duration (s)':<12} | {'Steps':<6}\n"
    print(header + "-" * len(header))

    for i in range(TEST_EPISODES):
        env.reset()
        done = False

        while not done:
            state = env.get_state(agent)
            if agent_type == "dqn":
                action = agent.select_action(state, env.get_valid_actions())
                _, _, done, _ = env.step(action)
                action = int(action) 
            else:
                action = agent.select_action(state) 
                _, done = env.step(action)

        print(f"{(i+1):>6}/{TEST_EPISODES} | {env.get_score():<12} | {env.get_duration():<12.4f} | {env.get_step_count():<6}")
        agent.episode_final_scores.append(env.get_score())
        agent.episode_durations.append(env.get_duration())
        agent.episode_step_counts.append(env.get_step_count())

    test_statistics(agent, f"{agent_type}_test_results.txt")
    save_test_statistics(agent, f"{agent_type}_test_statistics.npz")


def main(agent_type) -> None:

    if agent_type == "td":
        env = Environment()
        weights_file = TD_WEIGHTS_PATH.with_suffix('.npz')
        if weights_file.exists():
            agent = TDAgent(
                learning_rate=0.0,
                epsilon=0.0,
                epsilon_min=0.0,
                epsilon_decay=1.0,
                dynamic_lr=False,
            )
            agent.load(TD_WEIGHTS_PATH)
            print(f"Loaded TD weights from {weights_file}")
        else:
            print(f"TD weights file not found at {weights_file}. Please train the TD agent first to generate the weights file.")
            sys.exit(1)

    elif agent_type == "expectimax":
        env = Environment()
        agent = ExpectimaxAgent(depth=EXPECTIMAX_DEPTH)

    elif agent_type == "dqn":
        env = Env2048()
        dqn_file = DQN_MODEL_PATH
        if dqn_file.exists():
            agent = DQNAgent(
                env,
                lr=0,
                gamma=0,
                eps_start=0,
                eps_end=0,
                eps_decay=0,
                tau=0
            )
            agent.load(dqn_file)
            print(f"Loaded DQN model from {dqn_file}")
        else:
            print(f"DQN model file not found at {dqn_file}. Please train the DQN agent first to generate the model file.")
            sys.exit(1)

    else:
        print(f"Unknown agent type: {agent_type}. Please specify 'td' or 'expectimax'.")
        sys.exit(1)

    test_loop(agent, agent_type, env)

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            agent_type = sys.argv[1]
            main(agent_type)
        else:
            print("Please specify an agent type.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nGame interrupted by user. Exiting gracefully.")
        sys.exit(0)