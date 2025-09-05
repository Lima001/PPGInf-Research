"""
Performs a statistical analysis on a directory of binary-formatted
result files (e.g., from multiple training folds). For every possible
outcome combination (e.g., 1/0/1 for 3 tasks), it calculates the
percentage of occurrences in each file. It then reports the mean and
standard deviation of these percentages across all files.

Result files can be generated using both format.py and merge_files.py scripts.

Example (analyzes outcomes for tasks 0, 1, and 2 across all files):
    python analysis.py ./results/ 0 1 2
"""
import argparse
import os
import glob
import math
from collections import defaultdict
from typing import List, Dict

def parse_line(line: str):
    """Parses a line and returns the filename and a list of binary values."""
    parts = line.strip().split('/')
    
    if len(parts) < 2: return None, []
    
    filename = parts[0]
    
    try:
        return filename, list(map(int, parts[1:]))
    
    except ValueError:
        return None, []

def get_matching_counts(file_path: str, conditions: Dict[int, int]) -> int:
    """Counts entries in a file that match a specific set of conditions."""
    count = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        
        for line in file:
            _, flags = parse_line(line)
            
            if not flags: 
                continue
            
            try:
                if all(flags[pos] == value for pos, value in conditions.items()):
                    count += 1
            except IndexError:
                continue
    
    return count

def generate_report(directory: str, positions: List[int], verbose: bool) -> None:
    """Generates a report with mean and standard deviation of outcome percentages."""
    input_files = glob.glob(os.path.join(directory, '*.txt'))
    
    if not input_files:
        print(f"No .txt files found in directory: {directory}")
        return

    num_files = len(input_files)
    num_pos = len(positions)
    total_combinations = 2**num_pos
    scenario_percentages = defaultdict(list)

    print(f"Processing {num_files} files...")
    
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            total_entries = sum(1 for line in f if parse_line(line)[0] is not None)

        if total_entries == 0:
            print(f"Warning: No valid entries found in {os.path.basename(file_path)}. Skipping.")
            continue

        for i in range(total_combinations):
            binary_string = format(i, f'0{num_pos}b')
            conditions = {positions[j]: int(binary_string[j]) for j in range(num_pos)}
            
            count = get_matching_counts(file_path, conditions)
            percentage = (count / total_entries) * 100
            scenario_percentages[binary_string].append(percentage)

    # Final Summary Report
    print("\n--- Statistical Summary of Outcomes (across all files) ---")
    header_parts = [f"Pos {p}" for p in positions]
    header = " | ".join(header_parts) + " || Mean (%) | Std. Dev. (%)"
    print(header)
    print("-" * len(header))

    for scenario_bits in sorted(scenario_percentages.keys()):
        percentages = scenario_percentages[scenario_bits]
        
        if not percentages: 
            continue
        
        mean_perc = np.mean(percentages)
        std_perc = np.std(percentages)

        row_bits = " | ".join(list(scenario_bits))
        print(f"{row_bits} || {mean_perc:>8.2f} | {std_perc:>13.2f}")

    # Verbose Report (if enabled)
    if verbose:
        print("\n--- Verbose Breakdown: Per-File Percentages ---")
        
        for scenario_bits in sorted(scenario_percentages.keys()):
            percentages = scenario_percentages[scenario_bits]
            print(f"\n- Scenario ({', '.join(map(str, positions))}) = ({', '.join(list(scenario_bits))}):")
            
            for i, file_path in enumerate(input_files):
                print(f"  - {os.path.basename(file_path):<30}: {percentages[i]:.2f}%")

def main():
    """Main function to parse arguments and run the report generation."""
    parser = argparse.ArgumentParser(description="Generate a statistical report of outcomes from a directory of files.")
    parser.add_argument("directory", help="Path to the directory containing the result files.")
    parser.add_argument("positions", nargs='+', type=int, help="0-based positions to analyze (e.g., '0 1 2').")
    parser.add_argument("--verbose", action="store_true", help="Display per-file percentages for each scenario.")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found at {args.directory}")
    else:
        generate_report(args.directory, args.positions, args.verbose)

if __name__ == "__main__":
    main()