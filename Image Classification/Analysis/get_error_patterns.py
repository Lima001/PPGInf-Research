"""
Processes a prediction log file from a multi-task classification model. 
It identifies all misclassifications (where prediction != ground truth)
and ranks them by frequency. This helps diagnose the most common
error patterns in the model's predictions.

It provides two modes for ranking:
  1. By raw frequency (default): Shows the most common errors.
  2. By normalized rate (--normalize): Highlights the most challenging scenarios by accounting for class distribution.

Input:
  - A log file where each line contains prediction and ground truth tuples: "image.jpg | pred: (1, 10) | gt: (1, 12) | conf: (0.9, 0.8)"
  - Text files mapping class indices to human-readable names, one for each task.
"""
import argparse
import ast
from collections import Counter

def load_mappings(mapping_files):
    """Loads class index to class name mappings from text files."""
    mappings = []
    for path in mapping_files:
        mapping = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                key, val = line.strip().split(maxsplit=1)
                mapping[int(key)] = val
        mappings.append(mapping)
    return mappings

def parse_tuple_from_string(s):
    """Safely parses a string representation of a tuple."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return ()

def process_log_file(input_file, mappings):
    """Processes a log file to count errors and ground truth frequencies."""
    errors = Counter()
    gt_counts = Counter()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split('|')]
            if len(parts) < 3 or 'pred:' not in parts[1] or 'gt:' not in parts[2]:
                continue

            pred_str = parts[1].split(':')[1].strip()
            gt_str = parts[2].split(':')[1].strip()
            pred = parse_tuple_from_string(pred_str)
            gt = parse_tuple_from_string(gt_str)

            if not pred or not gt: continue
            
            try:
                mapped_gt = tuple(m[v] for m, v in zip(mappings, gt))
                mapped_pred = tuple(m[v] for m, v in zip(mappings, pred))
                
                gt_counts[mapped_gt] += 1
                
                if mapped_pred != mapped_gt:
                    errors[(mapped_gt, mapped_pred)] += 1
            except KeyError:
                print(f"Warning: A label index was not found. Line: {line.strip()}")
                continue
    
    return errors, gt_counts

def print_by_count(errors):
    """Prints a ranked list of errors by raw frequency."""
    total_errors = sum(errors.values())
    print("\n--- Ranked Misclassification Errors (by Raw Count) ---")
    print(f"{'Rank':<5} {'Ground Truth':<50} {'Prediction':<50} {'Frequency':<15} {'Count':<5}")
    print("-" * 125)
    
    for i, ((gt, pred), count) in enumerate(errors.most_common(), 1):
        percent = (count / total_errors) * 100
        print(f"{i:<5} {str(gt):<50} {str(pred):<50} {f'{percent:.2f}%':<15} {count:<5}")

def print_by_rate(errors, gt_counts):
    """Calculates normalized error rates and prints a ranked list."""
    error_rates = []
    for (gt, pred), count in errors.items():
        total_gt = gt_counts.get(gt, 0)
        rate = (count / total_gt) if total_gt > 0 else 0
        error_rates.append({"gt": gt, "pred": pred, "count": count, "rate": rate})
    
    sorted_errors = sorted(error_rates, key=lambda x: x['rate'], reverse=True)
    
    print("\n--- Ranked Misclassification Error Rates (Normalized by Ground Truth Frequency) ---")
    print(f"{'Rank':<5} {'Ground Truth':<45} {'Prediction':<45} {'Error Rate':<15} {'Count':<5}")
    print("-" * 120)
    
    for i, error in enumerate(sorted_errors, 1):
        rate_percent = f"{error['rate']:.2%}"
        print(f"{i:<5} {str(error['gt']):<45} {str(error['pred']):<45} {rate_percent:<15} {error['count']:<5}")

def main():
    """Main function to orchestrate the error analysis."""
    parser = argparse.ArgumentParser(description="Rank classification errors from a prediction log file.")
    parser.add_argument('--input', type=str, required=True, help='Input log file.')
    parser.add_argument('--mappings', type=str, required=True, nargs='+', help='Path to mapping files.')
    
    # Add an optional flag to control the ranking method.
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='If set, rank errors by normalized rate (considers class distribution).'
    )
    args = parser.parse_args()

    mappings = load_mappings(args.mappings)
    errors, gt_counts = process_log_file(args.input, mappings)

    if not errors:
        print("No errors found in the log file.")
        return

    # Conditional logic to call the correct printing function.
    if args.normalize:
        print_by_rate(errors, gt_counts)
    else:
        print_by_count(errors)

if __name__ == '__main__':
    main()