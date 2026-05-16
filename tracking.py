import matplotlib.pyplot as plt
import numpy as np
import os

def plot_tracking(episode_durations, episode_final_scores, filename="tracking_data.png"):
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

    filename = os.path.join("tracking", filename)
    plt.savefig(filename)
    print(f"Tracking plot saved to {filename}")

    plt.show()

def save_tracking(episode_durations, episode_final_scores, filename="tracking_data.npz"):
    filename = os.path.join("tracking", filename)
    np.savez(filename, episode_durations=episode_durations, episode_final_scores=episode_final_scores)
    print(f"Tracking data saved to {filename}")