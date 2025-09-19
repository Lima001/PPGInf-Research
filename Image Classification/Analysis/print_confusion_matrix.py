"""
Processes a classification log file for a specific task. It computes a confusion matrix in memory and prints a
formatted, human-readable version to the console, using a mapping file to show class names instead of indices.
"""
import argparse
import ast
from collections import Counter

def load_mapping(mapping_file):
    """Loads a class index to class name mapping from a single text file."""
    mapping = {}
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Use maxsplit=1 to handle names with spaces
                key, val = line.strip().split(maxsplit=1)
                mapping[int(key)] = val
    except FileNotFoundError:
        print(f"Error: Mapping file not found at '{mapping_file}'")
        exit(1)
    except ValueError:
        print(f"Error: Invalid format in mapping file '{mapping_file}'. Each line must be 'index name'.")
        exit(1)
    return mapping

def parse_line(line):
    """Safely parses a log line into filename, prediction, and ground truth tuples."""
    try:
        parts = line.strip().split('|')
        filename = parts[0].strip()
        pred_str = parts[1].split('pred:')[1].strip()
        gt_str = parts[2].split('gt:')[1].strip()
        return filename, ast.literal_eval(pred_str), ast.literal_eval(gt_str)
    except Exception:
        return None, None, None

def build_confusion_counter(input_log, mapping, task_index):
    """Processes the log file to count GT/Prediction pairs using class names."""
    confusion_counts = Counter()
    class_names = set(mapping.values()) # Pre-populate with all possible names

    with open(input_log, 'r', encoding='utf-8') as f:
        for line in f:
            _, pred_tuple, gt_tuple = parse_line(line)

            if not all([pred_tuple, gt_tuple]) or task_index >= len(gt_tuple):
                continue

            try:
                gt_index = gt_tuple[task_index]
                pred_index = pred_tuple[task_index]
                
                # Convert indices to names using the mapping
                gt_name = mapping.get(gt_index, f"Unknown_Index_{gt_index}")
                pred_name = mapping.get(pred_index, f"Unknown_Index_{pred_index}")

                confusion_counts[(gt_name, pred_name)] += 1
                class_names.add(gt_name)
                class_names.add(pred_name)
            
            except IndexError:
                continue

    return confusion_counts, sorted(list(class_names))

def print_confusion_matrix(counts, class_names):
    """
    Prints a formatted, row-normalized confusion matrix to the console as percentages.
    """
    if not class_names:
        print("No data to display.")
        return

    # --- 1. Calculate row totals for normalization ---
    row_totals = Counter()
    for (gt_name, _), count in counts.items():
        row_totals[gt_name] += count

    # Determine column width for alignment
    header_width = max(len(name) for name in class_names) if class_names else 0
    data_width = 7  # Width to fit "100.0 %"

    # --- 2. Print Header ---
    print("\n--- Confusion Matrix (Row-Normalized %) ---")
    header = f"{'GT \\ Pred':<{header_width}} |"
    for name in class_names:
        header += f" {name:>{data_width}}"
    print(header)
    print("-" * len(header))

    # --- 3. Print Rows with Percentages ---
    for gt_name in class_names:
        row_str = f"{gt_name:<{header_width}} |"
        total = row_totals.get(gt_name, 0)
        
        for pred_name in class_names:
            count = counts.get((gt_name, pred_name), 0)
            
            # Calculate percentage for the cell
            percent = (count / total) * 100 if total > 0 else 0
            
            # Format the cell string
            cell_str = f"{percent:.1f}%"
            row_str += f" {cell_str:>{data_width}}"
            
        print(row_str)

def main():
    """Main function to parse arguments and run the process."""
    parser = argparse.ArgumentParser(description="Print a confusion matrix from a log file using class names.")
    parser.add_argument('--input', type=str, required=True, help='Input log file from test.py --report-per-image.')
    parser.add_argument('--mapping', type=str, required=True, help='Path to the mapping file for the specified task.')
    parser.add_argument('--task-index', type=int, required=True, help='The 0-based index of the task to analyze.')
    args = parser.parse_args()

    # Load the class names
    mapping = load_mapping(args.mapping)
    
    # Process the log to get counts and all class names
    counts, class_names = build_confusion_counter(args.input, mapping, args.task_index)

    # Print the resulting matrix
    print_confusion_matrix(counts, class_names)

if __name__ == '__main__':
    main()