import os
import numpy as np
import matplotlib.pyplot as plt

def plot_tracking(episode_step_counts, episode_final_scores, windows_count=20, filename="tracking_data.png"):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    window = len(episode_final_scores) // windows_count

    # Plot 1: Episode Durations (Number of steps)
    # Background raw data scatter
    axs[0].plot(episode_step_counts, marker='.', linestyle='none', color='tab:blue', alpha=0.3, zorder=1)
    
    # Explicitly plot prominent mean + std error bars for every 200 episodes
    for i in range(0, len(episode_step_counts), window):
        block = episode_step_counts[i:i+window]
        if len(block) > 0:
            mean = np.mean(block)
            std = np.std(block)
            axs[0].errorbar(
                i + len(block)//2, mean, yerr=std, 
                fmt='o', ms=6, color='black', 
                ecolor='navy', elinewidth=2, 
                capsize=4, capthick=2, zorder=5
            )

    axs[0].set_title('Episode Durations (Number of steps)')
    axs[0].set_ylabel('Steps')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Episode Scores
    axs[1].plot(episode_final_scores, marker='.', linestyle='none', color='tab:orange', alpha=0.3, zorder=1)

    # Plot mean value and std of the block of 200 episodes
    for i in range(0, len(episode_final_scores), window):
        block = episode_final_scores[i:i+window]
        if len(block) > 0:
            mean = np.mean(block)
            std = np.std(block)
            axs[1].errorbar(
                i + len(block)//2, mean, yerr=std, 
                fmt='o',
                ms=6,
                color='black',
                ecolor='darkred',
                elinewidth=2,
                capsize=4,
                capthick=2,
                zorder=5
            )

    axs[1].set_title('Episode Scores')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Score')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # Ensure directory exists before saving
    os.makedirs("tracking", exist_ok=True)
    filename = os.path.join("tracking", filename)
    plt.savefig(filename, dpi=150) # Added higher DPI for crisper images
    print(f"Tracking plot saved to {filename}")

    plt.show()

def save_tracking(episode_step_counts, episode_durations, episode_final_scores, filename="tracking_data.npz"):
    filename = os.path.join("tracking", filename)
    np.savez(filename, episode_step_counts=episode_step_counts, episode_durations=episode_durations, episode_final_scores=episode_final_scores)
    print(f"Tracking data saved to {filename}")

def compute_statistics(episode_step_counts, episode_durations, episode_final_scores):
    stats = {
        "step_counts": {
            "mean": np.mean(episode_step_counts),
            "std": np.std(episode_step_counts),
            "min": np.min(episode_step_counts),
            "max": np.max(episode_step_counts)
        },
        "durations": {
            "mean": np.mean(episode_durations),
            "std": np.std(episode_durations),
            "min": np.min(episode_durations),
            "max": np.max(episode_durations)
        },
        "final_scores": {
            "mean": np.mean(episode_final_scores),
            "std": np.std(episode_final_scores),
            "min": np.min(episode_final_scores),
            "max": np.max(episode_final_scores)
        }
    }
    return stats

def test_statistics(agent, filename="testing_statistics.txt"):
    stats = compute_statistics(agent.episode_step_counts, agent.episode_durations, agent.episode_final_scores)
    filename = os.path.join("tracking", filename)
    with open(filename, "w") as f:
        for category, metrics in stats.items():
            f.write(f"{category.capitalize()}:")
            for metric, value in metrics.items():
                f.write(f" {metric}: {value}")
            f.write("\n")

def save_test_statistics(agent, filename="testing_statistics.npz"):
    stats = compute_statistics(agent.episode_step_counts, agent.episode_durations, agent.episode_final_scores)
    filename = os.path.join("tracking", filename)
    np.savez(filename, **stats)
    print(f"Testing statistics saved to {filename}")