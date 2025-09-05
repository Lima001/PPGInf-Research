"""
Converts a detailed prediction log file into a simplified binary format.
Each line in the output file represents an image and contains a series
of binary flags (1 for correct, 0 for incorrect) for each task, separated by '/'.

Input Format (from test.py --report-per-image):
  image_001.jpg | pred: (1, 10, 22) | gt: (1, 12, 22) | ...
"""
import argparse
import ast

def process_data(input_file, output_file):
    """
    Processes a file with prediction and ground truth data and writes
    the results to a new file in a binary comparison format.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            
            try:
                parts = line.strip().split(' | ')
                filename = parts[0]
                pred_str = parts[1].replace('pred: ', '')
                gt_str = parts[2].replace('gt: ', '')

                pred_tuple = ast.literal_eval(pred_str)
                gt_tuple = ast.literal_eval(gt_str)

                results = ['1' if p == g else '0' for p, g in zip(pred_tuple, gt_tuple)]
                output_line = f"{filename}/{'/'.join(results)}\n"
                outfile.write(output_line)

            except (IndexError, ValueError, SyntaxError) as e:
                print(f"Skipping malformed line: '{line.strip()}' due to error: {e}")
    
    print("Processing complete.")

def main():
    """Main function to parse arguments and run the data processing."""
    parser = argparse.ArgumentParser(description='Process a prediction log into a binary format.')
    parser.add_argument('input_file', type=str, help='The path to the input text file.')
    parser.add_argument('output_file', type=str, help='The path to the output text file.')
    
    args = parser.parse_args()
    process_data(args.input_file, args.output_file)

if __name__ == '__main__':
    main()