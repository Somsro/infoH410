import os
import numpy as np
from scipy import stats

def compare_algorithms(npz_path_alg1, npz_path_alg2, output_filename="algo_comparison_tests.txt"):
    """
    Loads tracking data for two algorithms and performs independent Student's t-tests
    and Wilcoxon rank-sum (Mann-Whitney U) tests on their metrics.
    """
    data1 = np.load(npz_path_alg1)
    data2 = np.load(npz_path_alg2)
    
    categories = ["step_counts", "final_scores"]
    results = {}

    for cat in categories:
        # Extract the full history arrays for both algorithms
        arr1 = data1[cat]
        arr2 = data2[cat]
        
        # Student's t-test
        t_stat, t_pval = stats.ttest_ind(arr1, arr2, equal_var=False)
        
        # Wilcoxon Rank-Sum / Mann-Whitney U test
        wilc_stat, wilc_pval = stats.mannwhitneyu(arr1, arr2, alternative='two-sided')
        
        results[cat] = {
            "t_test": {"statistic": t_stat, "p_value": t_pval},
            "wilcoxon": {"statistic": wilc_stat, "p_value": wilc_pval},
            "mean_alg1": np.mean(arr1),
            "mean_alg2": np.mean(arr2)
        }

    # Print results to console and save them to a text file
    os.makedirs("tracking", exist_ok=True)
    output_path = os.path.join("tracking", output_filename)
    
    with open(output_path, "w") as f:
        header = f"=== Statistical Comparison: {os.path.basename(npz_path_alg1)} vs {os.path.basename(npz_path_alg2)} ===\n"
        print(header)
        f.write(header + "\n")
        
        for cat, tests in results.items():
            summary = (
                f"Category: {cat.upper()}\n"
                f"  Alg 1 Mean: {tests['mean_alg1']:.2f} | Alg 2 Mean: {tests['mean_alg2']:.2f}\n"
                f"  Student's t-test:\n"
                f"    - Statistic: {tests['t_test']['statistic']:.4f}\n"
                f"    - p-value:   {tests['t_test']['p_value']:.4e}\n"
                f"  Wilcoxon / Mann-Whitney U test:\n"
                f"    - Statistic: {tests['wilcoxon']['statistic']:.4f}\n"
                f"    - p-value:   {tests['wilcoxon']['p_value']:.4e}\n"
            )
            
            # Highlight statistical significance (alpha = 0.05)
            alpha = 0.05
            if tests['wilcoxon']['p_value'] < alpha:
                summary += "  => Conclusion: The difference is STATISTICALLY SIGNIFICANT (p < 0.05).\n"
            else:
                summary += "  => Conclusion: No statistically significant difference detected (p >= 0.05).\n"
                
            summary += "-" * 60 + "\n"
            print(summary)
            f.write(summary + "\n")
            
    print(f"Statistical report saved to {output_path}")

if __name__ == "__main__":
    compare_algorithms("tracking/expectimax_test_statistics.npz", "tracking/td_test_statistics.npz", output_filename="expectimax_vs_td_comparison.txt")
    compare_algorithms("tracking/expectimax_test_statistics.npz", "tracking/dqn_test_statistics.npz", output_filename="expectimax_vs_dqn_comparison.txt")
    compare_algorithms("tracking/td_test_statistics.npz", "tracking/dqn_test_statistics.npz", output_filename="td_vs_dqn_comparison.txt")