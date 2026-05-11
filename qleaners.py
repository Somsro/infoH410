from board2048_ext import Board
import numpy as np

class Qlearner:
    """A Q-learning agent"""

    def __init__(
        self,
        action_size,
        state_size,
        learning_rate=0.0,
        gamma=0.98,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):
        self.action_size = action_size
        self.state_size = state_size

        # initialize the Q-table: (State x Agent Action)
        self.qtable = np.zeros((self.state_size, self.action_size))

        # define learning rate:
        if learning_rate == 0.0:
            self.dynamic_lr = True
            self.action_counter = np.zeros((self.state_size, self.action_size))
        else:
            self.dynamic_lr = False
            self.learning_rate = learning_rate

        # discount factor:
        self.gamma = gamma

        # Exploration parameters:
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # tracking rewards/progress:
        self.rewards_this_episode = []  # during an episode, save every time step's reward
        self.episode_total_rewards = []  # each episode, sum the rewards, possibly with a discount factor
        self.average_episode_total_rewards = []  # the average (discounted) episode reward to indicate progress

        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def reset_agent(self):
        self.qtable = np.zeros((self.state_size, self.action_size))

    def select_greedy(self, state):
        # np.argmax(self.qtable[state]) will select first entry if two or more Q-values are equal, but we want true randomness:
        return np.random.choice(np.flatnonzero(np.isclose(self.qtable[state], self.qtable[state].max()))) #select_greedy(state) -> action

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            #action = random.randrange(self.action_size) # action is an int
            action = np.random.randint(self.action_size)
        else:
            action = self.select_greedy(state)
        self.state_history.append(state)
        self.action_history.append(action)
        #!!!
        # print(self.action_history)
        return action

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def update(self, state, action, new_state, reward, done, update_epsilon=True):

        if not self.dynamic_lr:
            lr = self.learning_rate
        else:
            self.action_counter[state, action] += 1
            lr = 1 / self.action_counter[state, action]

        # Q(s,a) <-- Q(s,a) + learning_rate [R + gamma * max_a' Q(s',a') - Q(s,a)]
        # For our stateless, one-shot game, the update simplifies to: Q(s,a) <-- Q(s,a) + lr * [R - Q(s,a)]
        self.qtable[state, action] += lr * (reward + (not done) * self.gamma * np.max(self.qtable[new_state]) - self.qtable[state, action])

        self.rewards_this_episode.append(reward)
        #!!!
        self.reward_history.append(reward)
        #print("state ", state)
        
        if done:
            # track total reward:
            episode_reward = self._calculate_episode_reward(self.rewards_this_episode, discount=False)
            self.episode_total_rewards.append(episode_reward)

            k = len(self.average_episode_total_rewards) + 1  # amount of episodes that have passed
            self._calculate_average_episode_reward(k, episode_reward)

            if update_epsilon:
                self.update_epsilon()
                self.rewards_this_episode = [] #we reset here since if update_epsilon -> new episode

            

    def _calculate_episode_reward(self, rewards_this_episode, discount=False):
        if discount:
            return sum([(self.gamma**i) * reward for i, reward in enumerate(rewards_this_episode)])
        return sum(rewards_this_episode)

    def _calculate_average_episode_reward(self, k, episode_reward):
        if k > 1:  # running average is more efficient:
            average_episode_reward = (1 - 1 / k) * self.average_episode_total_rewards[-1] + episode_reward / k
        else:
            average_episode_reward = episode_reward
        self.average_episode_total_rewards.append(average_episode_reward)

    def print_rewards(self, episode, print_epsilon=True, print_q_table=True):
        print("Total (discounted) reward of this episode: ", self.episode_total_rewards[episode])
        print("All rewards: ", self.episode_total_rewards)
        print("Epsilon:", self.epsilon) if print_epsilon else None
        print("Q-table: ", self.qtable) if print_q_table else None


def q_learners(env_type, run):

    # Number of trial runs
    num_trials = 100
    num_episodes = 1000

    for trial in range(num_trials):
        # Initialize environment and agents for Independent Learners (ILs)
        

        ql_agent = Qlearner(
            action_size=4,
            state_size=65536,
            learning_rate=0.0,  # dynamic learning rate
            gamma=0.98,  # does not matter for stateless one-shot game
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.99
        )

        for episode in range(num_episodes):
            done = False

            # Agents take actions
            action = ql_agent.select_action(state)

            # Step in the environment
            observations, rewards, dones, _, infos = env.step(action)

            # Update Q-tables
            ql_agent.update(state, action, state, rewards, dones)