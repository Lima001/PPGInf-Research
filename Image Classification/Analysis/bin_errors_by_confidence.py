"""
Analyzes a detailed prediction log to find misclassifications for a specific task. 
It then groups these errors into bins based on the model's confidence score in its incorrect prediction. 
The script generates a summary table of these bins. An optional verbose mode lists every 
misclassified image file within each bin.

Input:
  - A log file where each line contains prediction, ground truth, and confidence tuples: 
    "image.jpg | pred: (1, 10) | gt: (1, 12) | conf: (0.9, 0.8)"
"""
import argparse
import ast
from pathlib import Path

def parse_tuple_from_string(s):
    """Safely parses a string representation of a tuple."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return ()

def process_log_and_bin_errors(input_file, task_index, num_bins):
    """Reads a log file and sorts misclassified image names into bins based on confidence."""
    binned_filenames = [[] for _ in range(num_bins)]
    total_errors = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split('|')]
            if len(parts) < 4 or 'pred:' not in parts[1] or 'gt:' not in parts[2] or 'conf:' not in parts[3]:
                continue

            img_name = parts[0].strip()
            
            pred = parse_tuple_from_string(parts[1].split(':')[1].strip())
            gt = parse_tuple_from_string(parts[2].split(':')[1].strip())
            conf = parse_tuple_from_string(parts[3].split(':')[1].strip())

            if not all([pred, gt, conf]) or task_index >= len(pred):
                continue

            if pred[task_index] != gt[task_index]:
                total_errors += 1
                confidence = max(0.0, min(1.0, conf[task_index]))
                bin_index = min(int(confidence * num_bins), num_bins - 1)
                binned_filenames[bin_index].append(img_name)

    return binned_filenames, total_errors

def print_binned_report(binned_errors, total_errors, num_bins, verbose):
    """Prints a summary table and optionally lists filenames in each bin."""
    print(f"\n--- Confidence Binning Report (Total Errors: {total_errors}) ---")
    print(f"{'Confidence Bin':<18} {'Error Count':<15} {'% of Total Errors':<20}")
    print("-" * 55)

    bin_width = 1.0 / num_bins
    
    for i, image_list in enumerate(binned_errors):
        low, high = i * bin_width, (i + 1) * bin_width
        count = len(image_list)
        percent = (count / total_errors * 100) if total_errors > 0 else 0.0
        bin_label = f"[{low:.2f} - {high:.2f}]"
        
        print(f"{bin_label:<18} {count:<15} {f'{percent:.2f}%':<20}")

    if verbose:
        print("\n--- Verbose Report: Filenames per Bin ---")
        for i, image_list in enumerate(binned_errors):
            if not image_list:
                continue
            low, high = i * bin_width, (i + 1) * bin_width
            print(f"\nBin [{low:.2f} - {high:.2f}] ({len(image_list)} errors):")
            for filename in image_list:
                print(f"  - {filename}")

def main():
    """Main function to orchestrate the error binning process."""
    parser = argparse.ArgumentParser(description="Generate a text report of errors binned by confidence score.")
    parser.add_argument('--input', type=str, required=True, help='Prediction log file from test.py.')
    parser.add_argument('--task-index', type=int, required=True, help='The 0-based index of the task to analyze.')
    parser.add_argument('--bins', type=int, default=10, help='Number of confidence bins.')
    parser.add_argument('--verbose', action='store_true', help='If set, print all filenames within each bin.')
    args = parser.parse_args()

    binned_filenames, total_errors = process_log_and_bin_errors(args.input, args.task_index, args.bins)

    if total_errors == 0:
        print(f"No errors found for task at index {args.task_index}.")
        return

    print_binned_report(binned_filenames, total_errors, args.bins, args.verbose)

if __name__ == '__main__':
    main()