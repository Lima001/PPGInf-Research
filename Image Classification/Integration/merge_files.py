"""
Merges multiple binary-formatted result files (format.py outputs) into a single file.
It verifies that all input files have the same number of lines and that
the filenames on each corresponding line match before merging. This is
useful for creating a combined result vector from different models or runs.
"""
import argparse

def merge_files(input_files, output_file):
    """
    Merges results from multiple input files into a single output file,
    verifying line counts and filenames.
    """
    
    if not input_files:
        print("Error: No input files provided.")
        return

    file_handles = []
    
    try:
        file_handles = [open(f, 'r', encoding='utf-8') for f in input_files]
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            line_number = 0
            while True:
                line_number += 1
                lines = [fh.readline().strip() for fh in file_handles]
                
                is_eof = [not line for line in lines]
                if all(is_eof):
                    break
                
                if any(is_eof):
                    print(f"Error: Files have mismatched line counts. Mismatch detected around line {line_number}.")
                    return

                filenames = [line.split('/')[0] for line in lines]
                if not all(f == filenames[0] for f in filenames):
                    print(f"Error: Filename mismatch on line {line_number}.")
                    return

                merged_results = []
                for line in lines:
                    results = line.split('/')[1:]
                    merged_results.extend(results)
                
                output_line = f"{filenames[0]}/{'/'.join(merged_results)}\n"
                outfile.write(output_line)
        
        print(f"Successfully merged {len(input_files)} files into '{output_file}'.")

    except FileNotFoundError as e:
        print(f"Error: An input file was not found: {e}")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    finally:
        for fh in file_handles:
            if fh:
                fh.close()

def main():
    """Main function to parse arguments and run the file merging process."""
    parser = argparse.ArgumentParser(description='Merge results from multiple files with matching filenames.')
    parser.add_argument('output_file', type=str, help='The path to the output text file.')
    parser.add_argument('input_files', nargs='+', help='A list of input text files to merge.')
    
    args = parser.parse_args()
    merge_files(args.input_files, args.output_file)

if __name__ == '__main__':
    main()