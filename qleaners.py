import numpy as np
from pathlib import Path

class Qlearner:
    """A Q-learning agent"""

    def __init__(
        self,
        action_size,
        state_size,
        learning_rate=0.1,
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
        self.episode_final_scores = []  # keep track of the final score for each episodes (episode_final_scores[i] = final score of episode i)
        self.average_episode_final_scores = []  # the average (discounted) episode final scores to indicate progress

        self.state_history = []
        self.action_history = []

    def reset_agent(self):
        self.qtable = np.zeros((self.state_size, self.action_size))

    def save_qtable(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, self.qtable)

    def load_qtable(self, filepath):
        filepath = Path(filepath)
        self.qtable = np.load(filepath, allow_pickle=False)
        if self.qtable.shape != (self.state_size, self.action_size):
            raise ValueError(
                f"Loaded qtable has shape {self.qtable.shape}, expected {(self.state_size, self.action_size)}"
            )

    def select_greedy(self, state, valid_moves):
        # np.argmax(self.qtable[state]) will select first entry if two or more Q-values are equal, but we want true randomness so we select randomly among the best Q-values:
        q_vals = self.qtable[state, valid_moves] # get the Q-values of the valid moves
        best_q_vals = np.flatnonzero(np.isclose(q_vals, q_vals.max())) # get the indice(s) of the valid moves that have the best Q-value
        return valid_moves[np.random.choice(best_q_vals)] #select_greedy(state) -> action

    def select_action(self, state, valid_moves):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(valid_moves) #explore: select a random valid move
        else:
            action = self.select_greedy(state, valid_moves) #exploit: select the best valid move according to the Q-table

        self.state_history.append(state)
        self.action_history.append(action)
        return action

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def update(self, state, action, new_state, reward, score, done, next_valid_actions, update_epsilon=True):

        if not self.dynamic_lr:
            lr = self.learning_rate
        else:
            self.action_counter[state, action] += 1
            lr = 1 / self.action_counter[state, action]

        #Update Q-table:
        target = reward
        if not done and next_valid_actions:
            target += self.gamma * np.max(self.qtable[new_state, next_valid_actions])

        # Q(s,a) <-- Q(s,a) + learning_rate [R + gamma * max_a' Q(s',a') - Q(s,a)]  Note : max_a' Q(s',a') = best next possible action so we need to compare on valid actions
        self.qtable[state, action] += lr * (target - self.qtable[state, action])

        self.rewards_this_episode.append(reward)
        
        if done:
            # track total score:
            self.episode_final_scores.append(score) #We track the total score for each episode (episode_final_scores[i] = final score of episode i)

            episodes_count = len(self.average_episode_final_scores) + 1  # amount of episodes that have passed
            self._calculate_average_episode_score(episodes_count, score)

            self.rewards_this_episode = [] #we reset here since if update_epsilon -> new episode
            self.state_history = []
            self.action_history = []

            if update_epsilon:
                self.update_epsilon()



    def _calculate_average_episode_score(self, k, score):
        if k > 1:  # running average is more efficient:
            average_episode_score = (1 - 1 / k) * self.average_episode_final_scores[-1] + score / k
        else:
            average_episode_score = score
        self.average_episode_final_scores.append(average_episode_score)

    def print_scores(self, episode, print_epsilon=True, print_q_table=True):
        print("Total score of this episode: ", self.episode_final_scores[episode])
        print("All scores: ", self.episode_final_scores)
        print("Epsilon:", self.epsilon) if print_epsilon else None
        print("Q-table: ", self.qtable) if print_q_table else None