"""
Analyzes a detailed prediction log to visualize the model's confidence distribution. 
It generates histograms for the confidence scores of both correct and incorrect predictions for a specified task. 
This is useful for diagnosing model calibration issues (e.g., over/under-confidence).

Input:
  - A log file where each line contains prediction, ground truth, and confidence tuples:
    "image.jpg | pred: (1, 10) | gt: (1, 12) | conf: (0.9, 0.8)"
"""
import argparse
import ast
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def parse_line(line):
    """Safely parses a log line into prediction, ground truth, and confidence tuples."""
    try:
        parts = line.strip().split('|')
        pred_str = parts[1].split('pred:')[1].strip()
        gt_str = parts[2].split('gt:')[1].strip()
        conf_str = parts[3].split('conf:')[1].strip()
        return ast.literal_eval(pred_str), ast.literal_eval(gt_str), ast.literal_eval(conf_str)
    
    except Exception:
        return None, None, None

def accumulate_confidence_values(file_path, task_index):
    """Reads a log file and extracts confidence values for correct and incorrect predictions."""
    correct_confidences, incorrect_confidences = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            pred, gt, conf = parse_line(line)
            if pred is None or task_index >= len(pred):
                continue
            
            confidence_value = conf[task_index]
            if pred[task_index] == gt[task_index]:
                correct_confidences.append(confidence_value)
            else:
                incorrect_confidences.append(confidence_value)
    
    return correct_confidences, incorrect_confidences

def plot_and_save(correct_conf, incorrect_conf, num_bins, output_path, task_index):
    """Generates and saves a PDF with three confidence histogram plots."""
    with PdfPages(output_path) as pdf:
        bins = np.linspace(0, 1, num_bins + 1)
        
        # Plot 1: Correct Predictions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(correct_conf, bins=bins, color='#0072B2', alpha=0.7, edgecolor='black', density=True)
        ax.set_title(f'Confidence Distribution for Correct Predictions (Task Index {task_index})')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Probability Density')
        pdf.savefig(fig)
        plt.close(fig)
        
        # Plot 2: Incorrect Predictions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(incorrect_conf, bins=bins, color='#D55E00', alpha=0.7, edgecolor='black', density=True)
        ax.set_title(f'Confidence Distribution for Incorrect Predictions (Task Index {task_index})')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Probability Density')
        pdf.savefig(fig)
        plt.close(fig)

        # Plot 3: Overlay
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(correct_conf, bins=bins, label='Correct', color='#0072B2', alpha=0.6, density=True)
        ax.hist(incorrect_conf, bins=bins, label='Incorrect', color='#D55E00', alpha=0.6, density=True)
        ax.set_title(f'Overlay of Confidence Distributions (Task Index {task_index})')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Probability Density')
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"Histogram plots saved to '{output_path}'.")

def main():
    """Main function to orchestrate histogram generation."""
    parser = argparse.ArgumentParser(description="Generate confidence histograms from a prediction log file.")
    parser.add_argument('--input_file', type=str, required=True, help='Input log file from test.py.')
    parser.add_argument('--task_index', type=int, required=True, help='The 0-based index of the task to analyze.')
    parser.add_argument('--bins', type=int, default=20, help='Number of bins for the histogram.')
    parser.add_argument('--output', '-o', type=str, default="confidence_report.pdf", help='Output PDF file name.')
    args = parser.parse_args()

    correct, incorrect = accumulate_confidence_values(args.input, args.task_index)
    if not correct and not incorrect:
        print("No valid data found in the log file for the specified task index.")
        return

    plot_and_save(correct, incorrect, args.bins, args.output, args.task_index)

if __name__ == "__main__":
    main()