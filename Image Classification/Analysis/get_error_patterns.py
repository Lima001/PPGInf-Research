"""
Processes a prediction log file from a multi-task classification model. 
It identifies all misclassifications (where prediction != ground truth)
and ranks them by frequency. This helps diagnose the most common
error patterns in the model's predictions.

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
    """Processes a log file to count and map prediction errors."""
    errors = Counter()
    total_errors = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split('|')]
            if len(parts) < 3 or 'pred:' not in parts[1] or 'gt:' not in parts[2]:
                continue

            pred_str = parts[1].split(':')[1].strip()
            gt_str = parts[2].split(':')[1].strip()
            
            pred = parse_tuple_from_string(pred_str)
            gt = parse_tuple_from_string(gt_str)

            if pred and gt and pred != gt:
                
                try:
                    mapped_pred = tuple(m[v] for m, v in zip(mappings, pred))
                    mapped_gt = tuple(m[v] for m, v in zip(mappings, gt))
                    errors[(mapped_pred, mapped_gt)] += 1
                    total_errors += 1
                
                except KeyError:
                    print(f"Warning: A label index was not found in the mapping files. Line: {line.strip()}")
                    continue
    
    return errors, total_errors

def print_ranked_errors(errors, total_errors):
    """Prints a ranked list of classification errors."""
    
    print("\n--- Ranked Misclassification Errors ---")
    print(f"{'Rank':<5} {'Ground Truth':<50} {'Prediction':<50} {'Frequency':<15} {'Count':<5}")
    print("-" * 87)
    
    for i, ((gt, pred), count) in enumerate(errors.most_common(), 1):
        percent = (count / total_errors) * 100
        print(f"{i:<5} {str(gt):<50} {str(pred):<50} {f'{percent:.2f}%':<15} {count:<5}")

def main():
    """Main function to orchestrate the error analysis."""
    
    parser = argparse.ArgumentParser(description="Rank classification errors from a prediction log file.")
    parser.add_argument('--input', type=str, required=True, help='Input log file with prediction, ground truth, and confidence.')
    parser.add_argument('--mappings', type=str, required=True, nargs='+', help='Path to mapping files (one for each task, in order).')
    args = parser.parse_args()

    mappings = load_mappings(args.mappings)
    errors, total_errors = process_log_file(args.input, mappings)

    if total_errors == 0:
        print("No errors found in the log file.")
    else:
        print_ranked_errors(errors, total_errors)

if __name__ == '__main__':
    main()