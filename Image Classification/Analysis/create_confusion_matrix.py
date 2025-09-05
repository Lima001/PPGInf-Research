"""
Creates a directory structure that mimics a confusion matrix for a specific task using integer class indices. 
For each image, it creates a symbolic link in a nested directory corresponding to
'./<GroundTruth_Index>/<Predicted_Index>/'. 
This allows for easy visual inspection of specific error types.
"""
import argparse
import ast
import os
from pathlib import Path
from tqdm import tqdm

def parse_line(line: str) -> Tuple[str, tuple, tuple]:
    """Safely parses a log line into filename, prediction, and ground truth tuples."""
    try:
        parts = line.strip().split('|')
        
        filename = parts[0].strip()
        pred_str = parts[1].split('pred:')[1].strip()
        gt_str = parts[2].split('gt:')[1].strip()
        
        return filename, ast.literal_eval(pred_str), ast.literal_eval(gt_str)
    
    except Exception:
        return None, None, None

def create_confusion_matrix_structure(input_log, img_root, output_dir, task_index):
    """Processes the log file and creates the symlink structure using indices."""
    print(f"Analyzing task at index {task_index}...")
    link_count = 0
    error_count = 0

    with open(input_log, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Creating confusion matrix"):
        filename, pred_tuple, gt_tuple = parse_line(line)

        if not all([filename, pred_tuple, gt_tuple]) or task_index >= len(gt_tuple):
            continue

        try:
            gt_index = gt_tuple[task_index]
            pred_index = pred_tuple[task_index]
        
        except IndexError:
            # This handles cases where a tuple might be shorter than expected
            continue

        # Define paths using integer indices
        target_dir = output_dir / str(gt_index) / str(pred_index)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        source_path = img_root / filename
        dest_path = target_dir / Path(filename).name

        # Create symlink if the source file exists and the link doesn't already
        if source_path.exists() and not dest_path.lexists():
            try:
                os.symlink(source_path.resolve(), dest_path)
                link_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error creating symlink for {source_path}: {e}")

    print("\nProcess Complete.")
    print(f"Created {link_count} new symbolic links in '{output_dir}'.")
    
    if error_count > 0:
        print(f"Encountered {error_count} errors during link creation.")

def main():
    """Main function to parse arguments and run the process."""
    parser = argparse.ArgumentParser(description="Build a confusion matrix directory structure with symlinks using class indices.")
    parser.add_argument('--input', type=str, required=True, help='Input log file from test.py --report-per-image.')
    parser.add_argument('--img-root', type=str, required=True, help='Root directory where original images are stored.')
    parser.add_argument('--output-dir', type=str, required=True, help='Base directory to build the confusion matrix structure.')
    parser.add_argument('--task-index', type=int, required=True, help='The 0-based index of the task to analyze.')
    args = parser.parse_args()

    try:
        create_confusion_matrix_structure(args.input, Path(args.img_root), Path(args.output_dir), args.task_index)
    except FileNotFoundError as e:
        print(f"Error: A required file or directory was not found. Please check your paths. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()