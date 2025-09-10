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
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional

def parse_line(line: str) -> Optional[Tuple[str, List[int]]]:
    """
    Parses a line and returns the filename and a list of binary values.
    Returns None if the line is malformed.
    """
    parts = line.strip().split('/')
    if len(parts) < 2:
        return None
    
    filename = parts[0]
    try:
        values = list(map(int, parts[1:]))
        return filename, values
    except ValueError:
        return None

def generate_report(directory: str, positions: List[int], verbose: bool) -> None:
    """
    Generates a report with mean and standard deviation of outcome percentages.
    This version is optimized to read each file only once.
    """
    input_files = sorted(glob.glob(os.path.join(directory, '*.txt')))
    
    if not input_files:
        print(f"No .txt files found in directory: {directory}")
        return

    num_files = len(input_files)
    num_pos = len(positions)
    total_combinations = 2**num_pos
    
    # This will store the list of percentages for each outcome scenario
    # Key: '010', Value: [9.5, 10.1, 9.8, ...]
    scenario_percentages = defaultdict(list)

    print(f"Processing {num_files} files...")
    
    # --- Main Optimized Loop ---
    # Read each file ONCE to gather all necessary counts.
    for file_path in input_files:
        counts = defaultdict(int)
        total_entries = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parsed = parse_line(line)
                if not parsed:
                    continue
                
                _, flags = parsed
                total_entries += 1
                
                try:
                    # Create the binary string key (e.g., '101') from the specified positions
                    outcome_key = "".join(str(flags[p]) for p in positions)
                    counts[outcome_key] += 1
                except IndexError:
                    # This happens if a line doesn't have enough flags for the requested positions
                    continue
        
        if total_entries == 0:
            print(f"Warning: No valid entries in {os.path.basename(file_path)}. Skipping.")
            # For every possible outcome, append 0% for this file to keep data consistent.
            for i in range(total_combinations):
                binary_string = format(i, f'0{num_pos}b')
                scenario_percentages[binary_string].append(0.0)
            continue

        # Calculate and store the percentage for every possible combination for this one file
        for i in range(total_combinations):
            binary_string = format(i, f'0{num_pos}b')
            count = counts[binary_string] # defaultdict returns 0 if key is not found
            percentage = (count / total_entries) * 100
            scenario_percentages[binary_string].append(percentage)

    # --- Final Summary Report ---
    print("\n--- Statistical Summary of Outcomes (across all files) ---")
    
    # **ALIGNMENT FIX**: Determine column widths based on the longest header text
    # This ensures that data rows align perfectly with the header.
    pos_col_width = max(len(f"Pos {p}") for p in positions)
    
    header_parts = [f"{f'Pos {p}':^{pos_col_width}}" for p in positions]
    header_base = " | ".join(header_parts)
    
    # Define a fixed width for the stats column for clean alignment
    stats_col_width = 18
    final_header = f"{header_base} || {'% (Std)':>{stats_col_width}}"
    
    print(final_header)
    print("-" * len(final_header))

    sorted_scenarios = sorted(scenario_percentages.keys())
    for scenario_bits in sorted_scenarios:
        percentages = scenario_percentages[scenario_bits]
        
        mean_perc = np.mean(percentages)
        std_perc = np.std(percentages)

        # **ALIGNMENT FIX**: Format each bit within the calculated column width
        row_parts = [f"{bit:^{pos_col_width}}" for bit in list(scenario_bits)]
        row_bits_aligned = " | ".join(row_parts)
        
        # Format the stats string first, then align it within the column
        stats_str = f"{mean_perc:.2f} ({std_perc:.2f})"
        print(f"{row_bits_aligned} || {stats_str:>{stats_col_width}}")

    # --- Verbose Report (if enabled) ---
    if verbose:
        print("\n--- Verbose Breakdown: Per-File Percentages ---")
        
        for scenario_bits in sorted_scenarios:
            percentages = scenario_percentages[scenario_bits]
            scenario_desc = ", ".join(f"Pos {p}={b}" for p, b in zip(positions, scenario_bits))
            print(f"\n- Scenario ({scenario_desc}):")
            
            for i, file_path in enumerate(input_files):
                print(f"  - {os.path.basename(file_path):<30}: {percentages[i]:.2f}%")

def main():
    """Main function to parse arguments and run the report generation."""
    parser = argparse.ArgumentParser(description="Generate a statistical report of outcomes from a directory of files.")
    parser.add_argument("directory", help="Path to the directory containing the result files.")
    parser.add_argument("positions", nargs='+', type=int, help="0-based positions to analyze (e.g., '0 1 2').")
    parser.add_argument("--verbose", "-v", action="store_true", help="Display per-file percentages for each scenario.")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found at {args.directory}")
    else:
        generate_report(args.directory, args.positions, args.verbose)

if __name__ == "__main__":
    main()
