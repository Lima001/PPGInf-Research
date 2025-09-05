# License Plate Quality Assessment Service:
#
# This script is a continuous service that assesses the quality of license plate images.
# It uses a YOLO-based classification model to sort images into 'accepted' or 'rejected'
# categories, based on the model's predictions. The service is designed to be
# highly efficient, handling a large number of pre-existing images at startup
# and continuously monitoring for new files.

import cv2
import torch
import time
import argparse
import threading
import logging
from pathlib import Path
from queue import Queue, Empty

from ultralytics import YOLO
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Valid file extensions for images
valid_extensions = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'}

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

def get_args():
    """Parses command-line arguments for the script."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',      type=str, required=True,    help='Directory with input images.')
    parser.add_argument('--output_dir',     type=str, required=True,    help='Directory to save quality assessment results.')
    parser.add_argument('--model_path',     type=str, required=True,    help='Path to the YOLO classification model file.')
    parser.add_argument('--device',         type=str, default='cuda:0', help="Device to run inference on (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument('--batch_size',     type=int, default=512,      help='Batch size for model inference.')
    parser.add_argument('--batch_timeout',  type=int, default=5,        help='Timeout in seconds for processing a partial batch.')
    
    return parser.parse_args()


def get_already_processed(output_dir):
    """
    Identifies images that have already been processed by checking for existing
    symbolic links in the 'accepted' and 'rejected' subdirectories.
    
    Args:
        output_dir (str): The directory where prediction files are saved.
        
    Returns:
        set: A set of base filenames that have been processed.
    """
    
    processed = set()
    output_path = Path(output_dir)
    
    if output_path.exists():
        for category in ['accepted', 'rejected']:
            cat_dir = output_path / category
            if cat_dir.exists():
                for link in cat_dir.glob("*"):
                    # Check if it's a symlink to a file
                    if link.is_symlink() and link.resolve().is_file():
                        processed.add(link.name)
    
    logger.info("Found %d already processed images in the output directory.", len(processed))
    
    return processed


def predict_images_batch(model, image_paths, accept_idx=(0, 1)):
    """
    Performs batch prediction on a list of images using a classification model.
    
    Args:
        model (YOLO): The loaded YOLO classification model.
        image_paths (list): A list of Path objects to the images.
        accept_idx (tuple): A tuple of class indices considered as 'accepted'.
    
    Returns:
        list: A list of tuples, each containing the original path,
              predicted class index, and a boolean indicating if it was accepted.
    """
    
    ims = []
    valid_paths = []
    
    for p in image_paths:
        try:
            im = cv2.imread(str(p))
            if im is not None:
                ims.append(im)
                valid_paths.append(p)
            else:
                logger.warning("Could not read image %s. Skipping.", p)
        except Exception:
            logger.error("An error occurred while reading file: %s", p.name, exc_info=True)
            
    if not ims:
        return []

    logger.debug("Starting YOLO inference on batch with %d images.", len(ims))
    
    results = model(ims, stream=False, verbose=False)
    predictions = []
    
    for p, r in zip(valid_paths, results):
        probs = r.probs
        if probs is None:
            logger.warning("Probs not available for %s", p.name)
            continue
        
        pred_idx = probs.top1
        accepted = pred_idx in accept_idx
        predictions.append((p, pred_idx, accepted))
    
    logger.info("Batch inference complete. Processed %d images.", len(valid_paths))
    
    return predictions


def save_predictions_batch(predictions, output_dir, processed_set, lock):
    """
    Saves the prediction results for a batch of images to the output directory
    by creating symbolic links and logging the quality score.
    
    Args:
        predictions (list): A list of prediction tuples from `predict_images_batch`.
        output_dir (str): The base directory to save the results.
        processed_set (set): The set of processed filenames to update.
        lock (threading.Lock):  The lock for thread-safe access to processed_set.
    """
    
    for image_path, pred_idx, accepted in predictions:
        category = 'accepted' if accepted else 'rejected'
        dest_dir = Path(output_dir) / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        link_path = dest_dir / image_path.name
        
        if not link_path.exists():
            try:
                link_path.symlink_to(image_path)
                logger.info("Created symbolic link for %s to %s. Quality score: %s", image_path.name, category, pred_idx)
            except OSError as e:
                logger.error("Failed to create symbolic link for %s: %s", image_path.name, e)
        else:
            logger.warning("Symbolic link for %s already exists. Skipping.", image_path.name)
        
        with lock: 
            processed_set.add(image_path.name)


def process_existing_files(input_dir, queue, processed):
    """
    Scans the input directory once at startup to queue any existing unprocessed images.
    
    Args:
        input_dir (str): The directory to scan for images.
        queue (Queue): The thread-safe queue for file paths.
        processed (set): A set of already processed filenames.
    """
    
    logger.info("Scanning for existing images recursively in the input directory: %s", input_dir)
    
    all_files = 0
    skipped = 0
    input_path = Path(input_dir)
    
    if input_path.exists():
        for fpath in input_path.rglob('*'):
            if not fpath.is_file():
                continue
            
            ext = fpath.suffix.lower()
            stem = fpath.name
            
            if ext in valid_extensions:
                all_files += 1
                if stem in processed:
                    skipped += 1
                    continue
                
                queue.put(fpath)
                logger.debug("Queued existing file: %s", fpath.name)

    logger.info("Initial scan complete. Total images found: %d, Skipped: %d, Queued: %d", all_files, skipped, all_files - skipped)


class LPQAHandler(FileSystemEventHandler):
    """Event handler for the watchdog observer. Puts new, valid image files into the processing queue."""
    
    def __init__(self, queue, processed, lock):
        self.queue = queue
        self.processed = processed
        self.lock = lock

    def on_created(self, event):
        file_path = Path(event.src_path)
        if event.is_directory or file_path.suffix.lower() not in valid_extensions:
            return
        
        stem = file_path.name
        with self.lock:
            if stem in self.processed:
                logger.info("Skipping new file %s, already processed.", file_path.name)
                return
        
        logger.info("New image file detected: %s. Adding to processing queue.", file_path.name)
        self.queue.put(file_path)


def batch_processor(queue, model, args, processed_set, stop_event, lock):
    """
    Worker function that runs in a separate thread.
    Pulls image paths from the queue and processes them in batches.
    
    Args:
        queue (Queue): The thread-safe queue for file paths.
        model (YOLO): The loaded YOLO model.
        args (Namespace): Parsed command-line arguments.
        processed_set (set): The set of processed filenames.
        stop_event (threading.Event): A signal to gracefully shut down.
        lock (threading.Lock):  The lock for thread-safe access to processed_set.
    """
    
    pending_paths = []
    last_process_time = time.time()
    waiting_message_printed = False

    logger.info("LPQA processor thread started.")

    while not stop_event.is_set() or not queue.empty() or pending_paths:
        try:
            while len(pending_paths) < args.batch_size and not stop_event.is_set():
                timeout = args.batch_timeout - (time.time() - last_process_time)
                path = queue.get(timeout=max(0.1, timeout))
                pending_paths.append(path)
                logger.debug("Added file to pending batch: %s", path.name)
        except Empty:
            pass

        now = time.time()
        
        if pending_paths and (len(pending_paths) >= args.batch_size or (now - last_process_time) >= args.batch_timeout or stop_event.is_set()):
            logger.info("Processing batch of %d images...", len(pending_paths))
            waiting_message_printed = False
            
            predictions = predict_images_batch(model, pending_paths, accept_idx=(0, 1))
            if predictions:
                save_predictions_batch(predictions, args.output_dir, processed_set, lock)
            
            pending_paths.clear()
            last_process_time = now
        
        else:
            if not pending_paths and not waiting_message_printed and (now - last_process_time) >= 10 and not stop_event.is_set():
                logger.info("Waiting for new images...")
                waiting_message_printed = True
                last_process_time = now


def main():
    """Main function to orchestrate the LPQA service."""
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    setup_logging('lpqa_service.log')

    processed_basenames = get_already_processed(args.output_dir)
    
    try:
        logger.info("Loading YOLO model from %s...", args.model_path)
        model = YOLO(args.model_path)
        device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
    except Exception:
        logger.error("Failed to load the YOLO model. Please check the model file path and integrity.", exc_info=True)
        return
    
    processing_queue = Queue()
    stop_event = threading.Event()
    lock = threading.Lock() 
    
    process_existing_files(args.input_dir, processing_queue, processed_basenames)
    
    processor_thread = threading.Thread(
        target=batch_processor,
        args=(processing_queue, model, args, processed_basenames, stop_event, lock),
        daemon=True
    )
    processor_thread.start()
    
    event_handler = LPQAHandler(processing_queue, processed_basenames, lock)
    observer = Observer()
    observer.schedule(event_handler, args.input_dir, recursive=True)
    observer.start()

    logger.info("LPQA service has started. Watching '%s' and its subdirectories for new images...", args.input_dir)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. LPQA service stop requested by user.")
        observer.stop()
        stop_event.set()
        logger.info("Waiting for the LPQA processor thread to finish its work...")
        observer.join()
        processor_thread.join()
    finally:
        logger.info("LPQA service has shut down completely.")
        
if __name__ == "__main__":
    start = time.time()
    main()
    logger.info("LPQA service runtime: %.2f seconds", time.time() - start)