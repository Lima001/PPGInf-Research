"""
A utility to count entries in a binary-formatted file that match a
specific set of conditions. It can also print the filenames of the
matching entries.


Example (finds lines where task 0 is correct and task 2 is incorrect):
    python matching.py formatted_report.txt "0=1" "2=0" --verbose
"""
import argparse

def parse_line(line):
    """Parses a line into a filename and a list of integer flags."""
    parts = line.strip().split('/')
    filename = parts[0]
    flags = list(map(int, parts[1:]))
    return filename, flags

def count_matching_entries(file_path, conditions, verbose):
    """Counts entries where specified flag positions match given values."""
    count = 0
    matching_filenames = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            filename, flags = parse_line(line)
            try:
                if all(flags[pos] == value for pos, value in conditions.items()):
                    count += 1
                    if verbose:
                        matching_filenames.append(filename)
            except IndexError:
                # This line has fewer flags than a position being checked, so it cannot match.
                continue
    
    print(f"Found {count} entries matching the specified conditions.")
    
    if verbose and matching_filenames:
        print("\nMatching filenames:")
        for name in matching_filenames:
            print(f"  - {name}")

def main():
    """Main function to parse arguments and run the matching logic."""
    parser = argparse.ArgumentParser(description="Count entries with specific flag values at given positions.")
    parser.add_argument("file", help="Path to the input text file.")
    parser.add_argument("conditions", nargs='+', help="Conditions in 'pos=value' format (e.g., '0=1 2=0').")
    parser.add_argument("--verbose", action="store_true", help="Display the filenames of matching entries.")
    args = parser.parse_args()
    
    try:
        conditions = {int(c.split('=')[0]): int(c.split('=')[1]) for c in args.conditions}
    except (ValueError, IndexError):
        print("Error: Invalid conditions format. Please use 'position=value' (e.g., '0=1').")
        return
    
    count_matching_entries(args.file, conditions, args.verbose)

if __name__ == "__main__":
    main()