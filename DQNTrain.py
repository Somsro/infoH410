from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import count
import torch
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from DQNutils import ReplayMemory, process_state, Transition
from DQNmodel import DQN
from DQNenv import Env2048

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 1000000
TAU = 0.005
LR = 1e-4


env = Env2048()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = 16

policy_net = DQN(n_actions=n_actions).to(device)
target_net = DQN(n_actions=n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

memory = ReplayMemory(100000)
steps_done = 0

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 200000
else:
    num_episodes = 50

def select_action(state, valid_actions):
    global steps_done
    
    
    if len(valid_actions) == 0:
        return torch.tensor([[0]], device=device, dtype=torch.long)
        
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:

        with torch.no_grad():
            q_values = policy_net(state)
            
            # We create a mask that will be added to the Q-values. The mask has -Inf for invalid actions and 0 for valid actions.
            mask = torch.full((1, 4), float('-inf'), device=device)
            
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
        return torch.tensor([[action]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_valid_mask_batch = torch.cat(batch.next_valid_mask)
    with torch.no_grad():
    
        # The network predicts Q-values for all actions in the next states. 
        # We then add the next_valid_mask to set the Q-values of invalid actions to -Inf, so that they won't be selected by max(1).values. 
        # Finally, we take the max over the action dimension to get the value of the best valid action in the next state. This is done only for non-final next states, for final next states the value is 0 as initialized above.
        next_q_values = target_net(non_final_next_states)
        
        # We add the next_valid_mask to the next_q_values to set the Q-values of invalid actions to -Inf, so that they won't be selected by max(1).values.
        masked_next_q_values = next_q_values + next_valid_mask_batch[non_final_mask]
        
        next_state_values[non_final_mask] = masked_next_q_values.max(1).values
        
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch


    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

MAX_TRAINING_TIME_SECONDS = 36000
start_time = time.time()
# Lists to keep track of progress
episode_durations = []
episode_final_scores = []
mean_score_200 = []
score_200 =[]
best_score = 0

pbar = tqdm(range(num_episodes), desc="Training Episodes")

# Wrap the range with tqdm for a progress bar
for i_episode in pbar:

    elapsed_time = time.time() - start_time
    if elapsed_time > MAX_TRAINING_TIME_SECONDS:
        print(f"\n Time Limit reached ({elapsed_time/60:.1f} min).")
        break # End training loop if time limit is exceeded
    
    # Initialize the environment and get its state
    observation, info = env.reset()
    state = process_state(observation, device)

    # Checkpoint every 200 episodes
    if (i_episode%200 == 0):

        if (i_episode != 0): 
            mean_score_200.append(sum(score_200)/len(score_200))
            score_200 = []
            
        checkpoint = {
                'episode': i_episode,
                'steps_done': steps_done,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
        torch.save(checkpoint, "dqn_2048_CHECK.pth")
    

    for t in count():

        valid_actions = [a for a in range(4) if env.unwrapped.board.is_move_valid(a)]
        action = select_action(state, valid_actions)
        next_observation, reward, terminated, truncated, _ = env.step(action.item())
                
        reward_tensor = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None

            next_valid_mask = torch.zeros((1, 4), device=device)
            
        else:
            next_state = process_state(next_observation, device)

            next_valid_actions = [a for a in range(4) if env.unwrapped.board.is_move_valid(a)]
            next_valid_mask = torch.full((1, 4), float('-inf'), device=device)
            for a in next_valid_actions:
                next_valid_mask[0, a] = 0.0

        # Store the transition in memory
        memory.push(state, action, next_state, reward_tensor, next_valid_mask)

        # Move to the next state
        state = next_state
        observation = next_observation

        # Optimize the model every 4 steps to balance training speed and stability       
        if steps_done % 4 == 0:
            optimize_model()
            
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            
            board_score = env.unwrapped.get_score(env.unwrapped.board)
            episode_final_scores.append(board_score)

            score_200.append(board_score)
            
            # Save the best model based on the highest score achieved so far
            if board_score > best_score:
                best_score = board_score
                torch.save(policy_net.state_dict(), "dqn_2048_BEST.pth")

            pbar.set_postfix({
                'Score': f"{board_score:.0f}",
                'Best': f"{best_score:.0f}",
            })
                
            break

last = {
                'episode': i_episode,
                'steps_done': steps_done,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
torch.save(last, "dqn_2048_LAST.pth")
print('Complete')


np.save("scores.npy", np.array(episode_final_scores))
np.save("duration.npy", np.array(episode_durations))

# ---------------------------------------------------------
# Final Plotting: Durations and Scores
# ---------------------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Episode Durations (Number of steps)
axs[0].plot(episode_durations, marker='.', linestyle='none', color='tab:blue', alpha=0.3)
axs[0].set_title('Episode Durations (Number of steps)')
axs[0].set_ylabel('Steps')
axs[0].grid(True)

# Plot 2: Episode Scores
axs[1].plot(episode_final_scores, marker='.', linestyle='none', color='tab:orange', alpha=0.3)

# Plotting mean score of every 200 epsiodes
window = 200
if len(episode_final_scores) >= window:
    means = np.convolve(episode_final_scores, np.ones(window)/window, mode='valid')
    axs[1].plot(range(window-1, len(episode_final_scores)), means, color='red', linewidth=2)

axs[1].set_title('Episode Scores')
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Score')
axs[1].grid(True)

plt.tight_layout()
plt.show()