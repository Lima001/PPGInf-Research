# This script is a continuous service that monitors the output log of the License Plate DB Matcher service 
# and filters the matched records. It separates entries into different categories based on correctness 
# (match/mismatch), and handles cases with unknown ground truth separately.

import argparse
import logging
import time
from pathlib import Path
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

def setup_logging(log_file):
    """
    Configures logging to output to both console and a file.
    
    Args:
        log_file (str): The full path to the log file.
    """
    
    log_dir = Path(log_file).parent
    if log_dir and not log_dir.exists():
        log_dir.mkdir(parents=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def normalize_plate(text):
    """
    Normalizes a license plate text string for comparison.
    
    Args:
        text (str): The license plate text to normalize.
        
    Returns:
        str: The normalized, lowercase text.
    """
    return text.strip().lower()

def get_base_filename(filename):
    """Extracts the base filename by removing the _lp suffix."""
    match = re.match(r'(.+)_lp\d+\.txt', filename)
    if match:
        return match.group(1)
    return filename.split('.')[0] # Fallback

def find_common_chars(s1, s2):
    """Finds the number of matching characters between two strings at the same position."""
    count = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            count += 1
    return count

class AnalysisHandler(FileSystemEventHandler):
    """
    A watchdog event handler that processes new lines appended to an input log file.
    This version assumes the input is sorted by base filename.
    
    Args:
        input_file_path (str): The path to the LPRM service's output log file.
        out_dir (str): The directory to save the analyzed output files.
        stop_event (threading.Event): A signal to gracefully shut down the service.
    """
    
    def __init__(self, input_file_path, out_dir, stop_event):
        self.input_file_path = Path(input_file_path)
        self.out_dir = Path(out_dir)
        self.stop_event = stop_event
        self.file_lock = threading.Lock()
        
        self.last_pos = 0
        self.current_group = []
        self.last_base_filename = None

        if not self.input_file_path.exists():
            logger.error("Input file not found at %s. Please check the path.", self.input_file_path)
            self.stop_event.set()
            return
            
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.open_writers = {}

        self.process_new_lines()

    def process_new_lines(self):
        """Processes any new lines appended to the input file, assuming they are sorted."""
        
        with self.file_lock:
            if not self.input_file_path.exists():
                return
            
            with self.input_file_path.open('r', encoding='utf-8') as f:
                f.seek(self.last_pos)
                new_lines = f.readlines()
                self.last_pos = f.tell()
        
        if new_lines:
            logger.info("Processing %d new lines from input file.", len(new_lines))
            
            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    filename, ocr, pmpr = line.split(',', 2)
                    base_filename = get_base_filename(filename)
                    
                    if self.last_base_filename and base_filename != self.last_base_filename:
                        # New group started, process the previous one
                        self._analyze_group(self.last_base_filename, self.current_group)
                        self.current_group = []

                    self.current_group.append({'filename': filename, 'ocr': ocr, 'pmpr': pmpr})
                    self.last_base_filename = base_filename
                    
                except ValueError:
                    logger.warning("Skipping malformed line: %s", line)

            # Process the last group at the end of the file
            if self.current_group:
                self._analyze_group(self.last_base_filename, self.current_group)
                self.current_group = [] # Clear the group after processing
                self.last_base_filename = None

    def _analyze_group(self, base_filename, entries):
        """Analyzes a single group of related entries."""
        if not entries:
            return
            
        logger.info("Analyzing group for base filename: %s", base_filename)
        
        # Check for UNKNOWN case (Rule I)
        if all(e['pmpr'].upper() == 'UNKNOWN' for e in entries):
            # Per your earlier request, the output for UNKNOWN is a single line
            output_line = f"{entries[0]['filename']},{entries[0]['ocr']},UNKNOWN"
            self._write_to_file('unknown.txt', output_line)
            return

        # Rule II: Find best OCR result(s)
        best_match_count = -1
        tied_entries = []
        database_result = None

        for entry in entries:
            pmpr = normalize_plate(entry['pmpr'])
            ocr = normalize_plate(entry['ocr'])
            database_result = entry['pmpr'] # Keep original for output
            match_count = find_common_chars(ocr, pmpr)

            if match_count > best_match_count:
                best_match_count = match_count
                tied_entries = [entry] # Start a new list of tied entries
            elif match_count == best_match_count:
                tied_entries.append(entry) # Add to the list of tied entries

        # Determine output file and content
        if len(tied_entries) > 1:
            output_file = f'tie-{best_match_count}.txt'
            for entry in tied_entries:
                output_line = f"{entry['filename']},{entry['ocr']},{entry['pmpr']}"
                self._write_to_file(output_file, output_line)
        else:
            best_entry = tied_entries[0]
            output_file = f'{best_match_count}.txt'
            output_line = f"{best_entry['filename']},{best_entry['ocr']},{best_entry['pmpr']}"
            self._write_to_file(output_file, output_line)
            
    def _write_to_file(self, filename, line):
        """Safely writes a line to a dynamically named output file."""
        output_path = self.out_dir / filename
        with self.file_lock:
            if filename not in self.open_writers:
                self.open_writers[filename] = output_path.open('a', encoding='utf-8', buffering=1)
            self.open_writers[filename].write(f"{line}\n")
        logger.info("Wrote line to %s.", filename)

    def on_modified(self, event):
        if not event.is_directory and Path(event.src_path).resolve() == self.input_file_path.resolve():
            self.process_new_lines()

    def stop_writers(self):
        """Closes all open output file handles."""
        logger.info("Closing all output file handles.")
        with self.file_lock:
            for f in self.open_writers.values():
                f.close()

def main():
    parser = argparse.ArgumentParser(description="Continuous service to filter LPRM results.")
    parser.add_argument('--input_file', required=True, help="Path to the input log file from LPRM.")
    parser.add_argument('--output_dir', required=True, help="Directory to save filtered output files.")
    parser.add_argument('--log_file', default='analysis.log', help="Path to the log file.")
    args = parser.parse_args()

    setup_logging(args.log_file)
    logger.info("Starting LPRM analysis service.")
    
    stop_event = threading.Event()

    handler = AnalysisHandler(args.input_file, args.output_dir, stop_event)
    
    observer = Observer()
    observer.schedule(handler, str(Path(args.input_file).parent), recursive=False)
    observer.start()

    logger.info("Watching %s for new entries...", args.input_file)
    
    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("LPRM analysis service stopped by user.")
        
    finally:
        observer.stop()
        stop_event.set()
        observer.join()
        
        # Before shutting down, process any remaining lines in the buffer to prevent data loss.
        logger.info("Performing final check for any remaining data...")
        handler.process_new_lines()
        if handler.current_group:
            logger.info("Processing final group before shutdown...")
            handler._analyze_group(handler.last_base_filename, handler.current_group)

        handler.stop_writers()
    
    logger.info("Service terminated.")

if __name__ == "__main__":
    start = time.time()
    main()
    logger.info("Service runtime: %.2f seconds", time.time() - start)