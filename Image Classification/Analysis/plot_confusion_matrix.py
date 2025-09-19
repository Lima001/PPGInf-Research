"""
Processes a classification log file for a specific task. It computes a confusion matrix and plots it to a vectorized pdf.
Similar to print_confusion_matrix.py, it uses a mapping file to show class names instead of indices.
"""
import argparse
import ast
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_mapping(mapping_file):
    """Loads a class index to class name mapping from a single text file."""
    mapping = {}
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                key, val = line.strip().split(maxsplit=1)
                mapping[int(key)] = val
    except FileNotFoundError:
        print(f"Error: Mapping file not found at '{mapping_file}'")
        exit(1)
    return mapping

def parse_line(line):
    """Safely parses a log line into filename, prediction, and ground truth tuples."""
    try:
        parts = line.strip().split('|')
        pred_str = parts[1].split('pred:')[1].strip()
        gt_str = parts[2].split('gt:')[1].strip()
        return ast.literal_eval(pred_str), ast.literal_eval(gt_str)
    except Exception:
        return None, None

def build_confusion_data(input_log, mapping, task_index):
    """Processes the log file and returns a pandas DataFrame of the confusion matrix."""
    counts = Counter()
    class_names = sorted(list(mapping.values()))
    
    with open(input_log, 'r', encoding='utf-8') as f:
        for line in f:
            pred_tuple, gt_tuple = parse_line(line)
            if not all([pred_tuple, gt_tuple]) or task_index >= len(gt_tuple):
                continue
            try:
                gt_name = mapping.get(gt_tuple[task_index])
                pred_name = mapping.get(pred_tuple[task_index])
                if gt_name and pred_name:
                    counts[(gt_name, pred_name)] += 1
            except (IndexError, KeyError):
                continue
    
    # Convert the counts to a pandas DataFrame, perfect for heatmaps
    df = pd.DataFrame(0, index=class_names, columns=class_names, dtype=int)
    for (gt, pred), count in counts.items():
        df.loc[gt, pred] = count
        
    return df

import matplotlib
# Add this import at the top

def plot_confusion_matrix(df, output_pdf_path):
    """
    Generates a well-dimensionalized confusion matrix plot, ensuring
    cell values are readable and the output is a true vector graphic.
    """
    # --- Add these two lines to prevent rasterization ---
    matplotlib.rcParams['agg.path.chunksize'] = 10000  # Increase complexity limit
    matplotlib.rcParams['pdf.fonttype'] = 42           # Use TrueType fonts

    if df.empty:
        print("Cannot plot an empty confusion matrix.")
        return

    # ... the rest of your function remains exactly the same ...

    # Normalize the matrix by row to get percentages
    df_normalized = df.div(df.sum(axis=1), axis=0).fillna(0)
    
    # Smart Sizing for Readability
    num_classes = len(df.columns)
    fig_size = max(10, num_classes * 0.6)
    
    if num_classes <= 10:   annot_size = 9
    elif num_classes <= 20: annot_size = 7
    elif num_classes <= 40: annot_size = 5
    elif num_classes <= 75: annot_size = 4
    else:                   annot_size = 3
    
    plt.figure(figsize=(fig_size, fig_size))

    # Create the Heatmap
    heatmap = sns.heatmap(
        df_normalized,
        annot=True,
        annot_kws={"size": annot_size},
        fmt=".1%",
        cmap="Blues",
        linewidths=0.5,
        linecolor="gray"
    )

    # Labels and Appearance
    title_fontsize = max(16, fig_size)
    label_fontsize = max(12, fig_size * 0.8)
    
    plt.title("Confusion Matrix (Normalized by Row)", fontsize=title_fontsize)
    plt.ylabel("True Label", fontsize=label_fontsize)
    plt.xlabel("Predicted Label", fontsize=label_fontsize)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Save to PDF
    try:
        plt.savefig(output_pdf_path, format="pdf", bbox_inches="tight")
        print(f"PDF saved to '{output_pdf_path}'")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate a confusion matrix plot from a log file.")
    parser.add_argument('--input', type=str, required=True, help='Input log file.')
    parser.add_argument('--mapping', type=str, required=True, help='Path to the mapping file for the task.')
    parser.add_argument('--task-index', type=int, required=True, help='0-based index of the task to analyze.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output PDF plot.')
    args = parser.parse_args()

    mapping = load_mapping(args.mapping)
    confusion_df = build_confusion_data(args.input, mapping, args.task_index)
    plot_confusion_matrix(confusion_df, args.output)

if __name__ == '__main__':
    main()