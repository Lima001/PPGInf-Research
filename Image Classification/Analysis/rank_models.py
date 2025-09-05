"""
Compares multiple models based on summary metric files. It performs
pairwise statistical tests to group models into performance tiers.
Models within the same tier are considered statistically equivalent in
performance for the given metric.


Input:
  - A directory containing .txt files. Each file represents a model and
    should contain summary metrics from multiple runs (e.g., from different
    folds), with one run per line, in the format:
    "Metric-Name-1: 0.1234, Metric-Name-2: 0.5678, ..."
"""
import argparse
import glob
import os
import re
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# --- Data Loading ---
def parse_metrics_from_line(line):
    """Dynamically extracts all 'key: value' metric pairs from a log line."""
    pattern = r"([\w\s().-]+?):\s*([\d.]+)"
    matches = re.findall(pattern, line)
    if not matches:
        raise ValueError("No metrics found in line.")
    
    metric_names = [match[0].strip() for match in matches]
    metric_values = [float(match[1]) for match in matches]
    return metric_names, metric_values

def load_model_scores(directory):
    """Loads scores from all .txt files, discovering metric names dynamically."""
    model_scores = defaultdict(lambda: defaultdict(list))
    metric_names_ordered = None

    for filepath in glob.glob(os.path.join(directory, "*.txt")):
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "r", encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    names, values = parse_metrics_from_line(line)
                    if metric_names_ordered is None:
                        metric_names_ordered = names
                    
                    if names != metric_names_ordered:
                        print(f"Warning: Inconsistent metric names in file {filepath}. Skipping line.")
                        continue
                        
                    for name, value in zip(names, values):
                        model_scores[model_name][name].append(value)
                except ValueError:
                    continue
    
    final_scores = {
        name: {metric: np.array(vals) for metric, vals in metrics.items()}
        for name, metrics in model_scores.items()
    }
    return final_scores, metric_names_ordered

# --- Statistical Analysis ---
def perform_pairwise_tests(scores, metric_name, alpha=0.05):
    """Performs Wilcoxon signed-rank test for all model pairs."""
    results = {}
    model_names = list(scores.keys())
    for m1, m2 in combinations(model_names, 2):
        vals1, vals2 = scores[m1][metric_name], scores[m2][metric_name]
        
        n = min(len(vals1), len(vals2))
        if n < 5: continue

        try:
            stat, p_value = wilcoxon(vals1[:n], vals2[:n])
        except ValueError:
            p_value = 1.0
        
        results[(m1, m2)] = {"p_value": p_value, "significant": p_value < alpha}
        results[(m2, m1)] = {"p_value": p_value, "significant": p_value < alpha}
    return results

def rank_models_into_tiers(scores, pairwise_results, metric_name):
    """Ranks models into tiers based on statistical significance."""
    mean_scores = {name: np.mean(vals[metric_name]) for name, vals in scores.items()}
    sorted_names = [name for name, _ in sorted(mean_scores.items(), key=lambda item: item[1], reverse=True)]
    
    tiers = []
    while sorted_names:
        current_tier = []
        # Start the tier with the current best model
        best_in_tier = sorted_names.pop(0)
        current_tier.append(best_in_tier)
        
        # Find other models that are NOT statistically different from the best in this tier
        remaining_models = []
        for other_model in sorted_names:
            if (best_in_tier, other_model) not in pairwise_results or not pairwise_results[(best_in_tier, other_model)]["significant"]:
                current_tier.append(other_model)
            else:
                remaining_models.append(other_model)
        
        tiers.append(current_tier)
        sorted_names = remaining_models
        
    return tiers, mean_scores

# --- Reporting ---
def print_tier_report(tiers, mean_scores, metric_name):
    """Generates and prints the tier-based ranking report."""
    print("\n" + "="*80)
    print(f"STATISTICAL PERFORMANCE TIERS FOR: {metric_name.upper()}")
    print("="*80)
    print("Models in the same tier are not statistically different from each other.")

    for i, tier in enumerate(tiers, 1):
        print(f"\n--- Tier {i} ---")
        # Sort models within the tier by their mean score for presentation
        sorted_tier = sorted(tier, key=lambda name: mean_scores[name], reverse=True)
        for name in sorted_tier:
            print(f"  {name:<40} | Mean Score: {mean_scores[name]:.4f}")

def main():
    """Main function to orchestrate the statistical ranking."""
    parser = argparse.ArgumentParser(description="Statistically rank models into performance tiers.")
    parser.add_argument('directory', type=str, help='Directory containing metric summary files from multiple models.')
    args = parser.parse_args()

    model_scores, metric_names = load_model_scores(args.directory)
    
    if not model_scores or not metric_names:
        print(f"No valid metric data found in '{args.directory}'.")
        return

    print(f"Discovered metrics: {', '.join(metric_names)}")

    for name in metric_names:
        pairwise_results = perform_pairwise_tests(model_scores, name)
        tiers, mean_scores = rank_models_into_tiers(model_scores, pairwise_results, name)
        print_tier_report(tiers, mean_scores, name)

if __name__ == '__main__':
    main()