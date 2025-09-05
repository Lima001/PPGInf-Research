# License Plate Database Matching Service
#
# This script is a continuous service that matches OCR (Optical Character Recognition)
# predictions of license plates against a large database of known records. It uses
# a watchdog to monitor for new OCR result files, which are then processed in batches
# and matched against an SQLite database for high-performance lookups.

import sqlite3
import os
import threading
import logging
import time
import argparse
from pathlib import Path
from queue import Queue, Empty

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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

class BatchLPRWatchdog(FileSystemEventHandler):
    """
    A watchdog event handler that queues new OCR files for processing and
    manages a worker thread to process batches.
    """
    def __init__(self, db_path, output_file, watch_dir, stop_event, batch_size, batch_timeout):
        self.db_path = db_path
        self.output_file = Path(output_file)
        self.watch_dir = Path(watch_dir)
        self.queue = Queue()
        self.lock = threading.Lock()
        self.stop_event = stop_event
        self.processed_files = set()
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        # Database connection is now handled in the worker thread

        self._load_processed_files()

        self._batch_worker_thread = threading.Thread(target=self._batch_worker, daemon=False)
        self._batch_worker_thread.start()
        
        self.process_existing()

    def _load_processed_files(self):
        """Loads already processed filenames from the output file."""
        if self.output_file.exists():
            logger.info("Loading already processed files from %s...", self.output_file)
            try:
                with self.output_file.open('r', encoding='utf-8') as f:
                    for line in f:
                        filename = line.split(',', 1)[0].strip()
                        self.processed_files.add(filename)
            except Exception:
                logger.error("Error loading processed files from %s.", self.output_file, exc_info=True)
            logger.info("Loaded %d processed files.", len(self.processed_files))


    def _batch_worker(self):
        """Worker thread function that processes files from the queue in batches."""
        logger.info("Batch processing worker thread started.")
        
        # Create the database connection and cursor inside the worker thread
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        pending = []
        last_process = time.time()
        waiting_message_printed = False

        while not self.stop_event.is_set() or not self.queue.empty() or pending:
            try:
                # Get files until a batch is full or timeout
                while len(pending) < self.batch_size and not self.stop_event.is_set():
                    item = self.queue.get(timeout=1)
                    pending.append(item)
            except Empty:
                pass

            now = time.time()
            if pending and (len(pending) >= self.batch_size or (now - last_process) >= self.batch_timeout or self.stop_event.is_set()):
                logger.info("Processing batch of %d files...", len(pending))
                waiting_message_printed = False
                # Pass the cursor to the processing function
                self.process_batch(pending, cursor)
                pending.clear()
                last_process = now
            else:
                if not pending and not waiting_message_printed and (now - last_process) >= 10 and not self.stop_event.is_set():
                    logger.info("Waiting for new images...")
                    waiting_message_printed = True
                    last_process = now
        
        # Close the connection when the worker thread finishes
        conn.close()
        logger.info("Batch processing worker thread finished and database connection closed.")

    # The process_batch function now takes the cursor as an argument
    def process_batch(self, files, cursor):
        """
        Processes a batch of OCR result files.
        
        It reads the OCR text, matches it against the database, and appends
        the results to the output file.
        
        Args:
            files (list): A list of file paths to process.
            cursor (sqlite3.Cursor): The database cursor for this thread.
        """
        output_lines = []
        new_processed = []
        
        filenames_db = []
        for lpr_file in files:
            with self.lock:
                if lpr_file.name in self.processed_files:
                    logger.info("Skipping already processed file: %s", lpr_file.name)
                    continue
            
            parts = lpr_file.stem.split('_lp')
            if len(parts) > 1:
                filename = parts[0] + '.jpg'
                filenames_db.append(filename)
            else:
                logger.warning("File %s does not follow the expected naming convention.", lpr_file.name)
                filenames_db.append(lpr_file.name)

        if not filenames_db:
            return

        placeholders = ','.join('?' for _ in filenames_db)
        query = f"SELECT filename, pmpr FROM records WHERE filename IN ({placeholders})"
        
        # Use the passed cursor for the query
        db_results = dict(cursor.execute(query, filenames_db).fetchall())

        for lpr_file in files:
            try:
                with lpr_file.open('r', encoding='utf-8') as f:
                    lines = f.readlines()
                if not lines:
                    logger.warning("Warning: %s is empty", lpr_file.name)
                    continue
                ocr = lines[0].strip()
            except Exception:
                logger.error("An error occurred while reading %s.", lpr_file.name, exc_info=True)
                continue
            
            parts = lpr_file.stem.split('_lp')
            db_lookup_name = parts[0] + '.jpg' if len(parts) > 1 else lpr_file.name
            
            pmpr = db_results.get(db_lookup_name, 'UNKNOWN')
            output_lines.append(f"{lpr_file.name},{ocr},{pmpr}\n")
            new_processed.append(lpr_file.name)

        with self.lock:
            with self.output_file.open('a', encoding='utf-8') as out_f:
                out_f.writelines(output_lines)
            self.processed_files.update(new_processed)

        logger.info("Batch completed: %d new files processed.", len(new_processed))

    def on_created(self, event):
        """Handles the 'on_created' file system event, queuing new, unprocessed `.txt` files."""
        src_path = Path(event.src_path)
        if event.is_directory or src_path.suffix.lower() != '.txt':
            return
        
        with self.lock:
            if src_path.name in self.processed_files:
                logger.info("Skipping new file %s, already processed.", src_path.name)
                return

        logger.info("New file detected: %s", src_path.name)
        self.queue.put(src_path)
    
    def process_existing(self):
        """Scans the watch directory once at startup to queue existing, unprocessed files."""
        existing_files = list(self.watch_dir.glob("*.txt"))
        unprocessed_files = []
        with self.lock:
            unprocessed_files = [f for f in existing_files if f.name not in self.processed_files]

        logger.info("Found %d existing .txt files.", len(existing_files))
        logger.info("%d unprocessed files to process.", len(unprocessed_files))
        
        for fpath in unprocessed_files:
            self.queue.put(fpath)

def main():
    """Main function to orchestrate the LPRM service."""
    parser = argparse.ArgumentParser(description="Efficient LPR matching with batch processing.")
    parser.add_argument('--db', required=True, help='Path to SQLite database')
    parser.add_argument('--watch_dir', required=True, help='Directory to watch for .txt files')
    parser.add_argument('--output_file', required=True, help='Path to output log file')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for OCR processing.')
    parser.add_argument('--batch_timeout', type=int, default=5, help='Timeout in seconds to process a partial batch.')
    args = parser.parse_args()

    setup_logging('lpdbm_service.log')
    db_path = Path(args.db)
    watch_dir = Path(args.watch_dir)
    output_file = Path(args.output_file)

    if not db_path.exists():
        logger.error("Error: Database file not found at %s. Please run db_creator.py first.", db_path)
        return

    stop_event = threading.Event()
    handler = BatchLPRWatchdog(str(db_path), str(output_file), str(watch_dir), stop_event, args.batch_size, args.batch_timeout)

    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=False)
    observer.start()

    logger.info("Watching %s for new .txt files...", watch_dir)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user, stopping...")
    finally:
        observer.stop()
        stop_event.set()
        observer.join()
        handler._batch_worker_thread.join()
        logger.info("LPRM service has shut down completely.")

if __name__ == "__main__":
    start = time.time()
    main()
    logger.info("LPRM service runtime: %.2f seconds", time.time() - start)
