import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
import random
import math
import torch.optim as optim
from tqdm import tqdm
from time import time
from DQNutils import ReplayMemory, Transition
from PARAMETERS import DQN_BATCH_SIZE, DQN_GAMMA, DQN_EPS_START, DQN_EPS_END, DQN_EPS_DECAY, DQN_TAU, DQN_LR, NUM_EPISODES, MAX_TRAINING_TIME_SECONDS, DQN_MODEL_PATH
from DQNutils import process_state


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        d = output_dim // 4
        # The model looks at the input with 4 different convolutional filters of sizes 1, 2, 3 and 4. Each filter produces d output channels.
        self.conv1 = nn.Conv2d(input_dim, d, 1, padding='same')
        self.conv2 = nn.Conv2d(input_dim, d, 2, padding='same')
        self.conv3 = nn.Conv2d(input_dim, d, 3, padding='same')
        self.conv4 = nn.Conv2d(input_dim, d, 4, padding='same')

    def forward(self, x):
        output1 = F.relu(self.conv1(x))
        output2 = F.relu(self.conv2(x))
        output3 = F.relu(self.conv3(x))
        output4 = F.relu(self.conv4(x))
        # We stack the outputs of the 4 convolutional filters along the channel dimension to get a tensor of shape (batch_size, output_dim, height, width).
        return torch.cat((output1, output2, output3, output4), dim=1)

class DQN(nn.Module):
    def __init__(self, n_actions=4):
        super(DQN, self).__init__()
        
        # 16 channels en entrée (ton one-hot encoding)
        self.conv1 = ConvBlock(16, 512)
        self.conv2 = ConvBlock(512, 512)
        
        # Le Flatten transforme les 512 filtres de taille 4x4 en une ligne de 8192 neurones
        self.dense1 = nn.Linear(8192, 512)
        self.dense2 = nn.Linear(512, n_actions)
    
    # How the model processes the input state to produce Q-values for each action
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.Flatten()(x)
        x = F.relu(self.dense1(x))
        return self.dense2(x)
    
class DQNAgent:
    def __init__(self, env, lr=DQN_LR, gamma=DQN_GAMMA, eps_start=DQN_EPS_START, eps_end=DQN_EPS_END, eps_decay=DQN_EPS_DECAY, tau=DQN_TAU):
        n_actions = env.action_space.n
        self.model = DQN(n_actions)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau

        self.policy_net = DQN(n_actions=n_actions).to(self.device)
        self.target_net = DQN(n_actions=n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.step_count = 0
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(10000)
        
        self.episode_final_scores = []
        self.episode_durations = []
        self.episode_step_counts = []


    def select_action(self, state, valid_actions):
        """Choose an action with epsilon-greedy exploration (restricted to valid moves)."""

        if len(valid_actions) == 0:
            return torch.tensor([[0]], device=self.device, dtype=torch.long)
            
        sample = random.random()
        if self.eps_decay == 0:
            eps_threshold = 0
        else:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.step_count / self.eps_decay)
        self.step_count += 1
        
        if sample > eps_threshold:

            with torch.no_grad():
                q_values = self.policy_net(state)
                
                # We create a mask that will be added to the Q-values. The mask has -Inf for invalid actions and 0 for valid actions.
                mask = torch.full((1, 4), float('-inf'), device=self.device)
                
                # Valid actions will have a mask value of 0, invalid actions will have a mask value of -Inf
                for a in valid_actions:
                    mask[0, a] = 0.0 
                    
                # Invalid actions will have a Q-value of -Inf, valid actions keep their original Q-values
                masked_q_values = q_values + mask
                
                # We select the action with the highest Q-value among the valid actions (the invalid actions will have -Inf and thus will never be selected)
                return masked_q_values.max(1)[1].view(1, 1)
        else:
            # Randomly select an action from the valid actions
            action = random.choice(valid_actions)
            return torch.tensor([[action]], device=self.device, dtype=torch.long)
        
    def optimize_model(self):
        if len(self.memory) < DQN_BATCH_SIZE:
            return
        transitions = self.memory.sample(DQN_BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(DQN_BATCH_SIZE, device=self.device)
        next_valid_mask_batch = torch.cat(batch.next_valid_mask)
        with torch.no_grad():
        
            # The network predicts Q-values for all actions in the next states. 
            # We then add the next_valid_mask to set the Q-values of invalid actions to -Inf, so that they won't be selected by max(1).values. 
            # Finally, we take the max over the action dimension to get the value of the best valid action in the next state. This is done only for non-final next states, for final next states the value is 0 as initialized above.
            next_q_values = self.target_net(non_final_next_states)
            
            # We add the next_valid_mask to the next_q_values to set the Q-values of invalid actions to -Inf, so that they won't be selected by max(1).values.
            masked_next_q_values = next_q_values + next_valid_mask_batch[non_final_mask]
            
            next_state_values[non_final_mask] = masked_next_q_values.max(1).values
            
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch


        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def load(self, path):
        """Load a saved network checkpoint from disk."""

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])

    def save(self, path, env, episode):
        """Save the current training state, including model weights and optimizer."""

        checkpoint = {
            'episode': episode,
            'steps_done': env.get_step_count(),
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)


def train_dqn(env):
    """Train a DQN agent for NUM_EPISODES (with time limit)
    return the trained agent."""

    agent = DQNAgent(env)

    pbar = tqdm(range(NUM_EPISODES), desc="Training Episodes")
    start_time = time()
    best_score = 0
    for i_episode in pbar:

        elapsed_time = time() - start_time
        if elapsed_time > MAX_TRAINING_TIME_SECONDS:
            print(f"\n Time Limit reached ({elapsed_time/60:.1f} min).")
            break # End training loop if time limit is exceeded
        
        # Initialize the environment and get its state
        observation, _ = env.reset()
        state = process_state(observation, agent.device)

        # Checkpoint every 200 episodes
        if (i_episode%200 == 0):
            agent.save(DQN_MODEL_PATH, env, i_episode)
        

        for t in count():

            valid_actions = [a for a in range(4) if env.unwrapped.board.is_move_valid(a)]
            action = agent.select_action(state, valid_actions)
            next_observation, reward, terminated, truncated = env.step(action.item())
                    
            reward_tensor = torch.tensor([reward], device=agent.device)
            done = terminated or truncated

            if terminated:
                next_state = None

                next_valid_mask = torch.zeros((1, 4), device=agent.device)
                
            else:
                next_state = process_state(next_observation, agent.device)

                next_valid_actions = [a for a in range(4) if env.unwrapped.board.is_move_valid(a)]
                next_valid_mask = torch.full((1, 4), float('-inf'), device=agent.device)
                for a in next_valid_actions:
                    next_valid_mask[0, a] = 0.0

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward_tensor, next_valid_mask)

            # Move to the next state
            state = next_state
            observation = next_observation

            # Optimize the model every 4 steps to balance training speed and stability       
            if env.get_step_count() % 4 == 0:
                agent.optimize_model()
                
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = agent.target_net.state_dict()
                policy_net_state_dict = agent.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*agent.tau + target_net_state_dict[key]*(1-agent.tau)
                agent.target_net.load_state_dict(target_net_state_dict)

            if done:
                agent.episode_step_counts.append(t + 1)
                agent.episode_durations.append(env.get_duration())
                
                board_score = env.unwrapped.get_score()
                agent.episode_final_scores.append(board_score)
                
                # Save the best model based on the highest score achieved so far
                if board_score > best_score:
                    best_score = board_score
                    torch.save(agent.policy_net.state_dict(), DQN_MODEL_PATH)

                pbar.set_postfix({
                    'Score': f"{board_score:.0f}",
                    'Best': f"{best_score:.0f}",
                })
                    
                break

    agent.save(DQN_MODEL_PATH, env, i_episode)

    return agent