# License Plate Recognition Service
#
# This script is a continuous service for license plate recognition (LPR).
# It is designed to work in tandem with a License Plate Detection (LPD) system,
# taking cropped license plate images and performing Optical Character
# Recognition (OCR) on them.
#
# The code is specifically designed to use the ParSeq model for OCR, a highly
# efficient Transformer-based architecture known for its accuracy in recognizing
# text in complex environments.

import os
import time
import threading
import argparse
import logging
from pathlib import Path
from queue import Queue, Empty

from PIL import Image
import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from strhub.data.module import SceneTextDataModule
from strhub.models.parseq.system import PARSeq as ModelClass

valid_extensions = {'.png', '.jpg', '.jpeg'}

logger = logging.getLogger(__name__)

def setup_logging(log_file):
    """
    Configures logging to output to both the console and a specified log file.
    
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
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the ParSeq model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing cropped license plate images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save OCR results as text files')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for OCR inference')
    parser.add_argument('--batch_timeout', type=int, default=5, help='Timeout in seconds to process a partial batch')
    parser.add_argument('--device', default='cuda', help='Device to run inference on (e.g., "cuda" or "cpu")')
    
    return parser.parse_args()

@torch.inference_mode()
def recognize_batch(model, img_transform, image_paths, device):
    """
    Performs text recognition on a batch of images using the loaded ParSeq model.
    
    Args:
        model (torch.nn.Module): The loaded ParSeq model.
        img_transform (callable): The image transformation function.
        image_paths (list): A list of file paths to the images.
        device (str): The device to run inference on.
        
    Returns:
        dict: A dictionary mapping image paths to their recognized text and confidence scores.
    """
    images = []
    valid_paths = []
    
    logger.debug("Recognizing batch with %d images...", len(image_paths))
    
    for p in image_paths:
        try:
            img = Image.open(p).convert('RGB')
            images.append(img_transform(img))
            valid_paths.append(p)
        except Exception:
            logger.error("An error occurred while opening image %s.", p.name, exc_info=True)

    if not images:
        return {}
    
    try:
        batch = torch.stack(images).to(device)
        probs = model(batch).softmax(-1)
        preds, confidences = model.tokenizer.decode(probs)
        results = {}
        for path, pred, conf in zip(valid_paths, preds, confidences):
            results[path] = (pred, conf.tolist()[:-1])
        return results
    except Exception:
        logger.error("An error occurred during model inference.", exc_info=True)
        return {}


def get_already_processed(output_dir):
    """
    Identifies files already processed by checking for existing output `.txt` files.
    
    Args:
        output_dir (str): The directory to check for processed files.
        
    Returns:
        set: A set of base filenames that have been processed.
    """
    
    processed = set()
    output_path = Path(output_dir)
    
    if output_path.exists():
        for f in output_path.glob("*.txt"):
            processed.add(f.stem)
            
    logger.info("Found %d already processed OCR files.", len(processed))
    
    return processed

class LPRHandler(FileSystemEventHandler):
    """
    Event handler for the watchdog observer. Puts new, valid image files into the processing queue.
    
    Args:
        queue (Queue): The thread-safe queue for file paths.
        processed (set): A set of already processed filenames.
    """
    
    def __init__(self, queue, processed, lock):
        self.queue = queue
        self.processed = processed
        self.lock = lock

    def on_created(self, event):
        file_path = Path(event.src_path)
        if event.is_directory or file_path.suffix.lower() not in valid_extensions:
            return
        
        stem = file_path.stem
        with self.lock:
            if stem in self.processed:
                logger.info("Skipping new file %s, already processed.", file_path.name)
                return
             
        logger.info("New image file detected: %s. Adding to processing queue.", file_path.name)
        self.queue.put(file_path)

def process_existing_files(input_dir, queue, processed):
    """
    Scans the input directory once at startup to queue any existing unprocessed images.
    
    Args:
        input_dir (str): The directory to scan.
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
            stem = fpath.stem
            
            if ext in valid_extensions:
                all_files += 1
                if stem in processed:
                    skipped += 1
                    logger.info("Skipping existing file %s, already processed.", fpath.name)
                    continue
                
                logger.info("Queued existing file for processing: %s", fpath.name)
                queue.put(fpath)
    
    logger.info("Initial scan complete. Total images found: %d, Skipped: %d, Queued: %d", all_files, skipped, all_files - skipped)

def batch_processor(queue, model, img_transform, output_dir, batch_size, device, batch_timeout, processed, stop_event, lock):
    """
    Worker function that runs in a separate thread. Pulls image paths from the queue and processes them in batches.
    
    Args:
        queue (Queue): The thread-safe queue for file paths.
        model (torch.nn.Module): The loaded ParSeq model.
        img_transform (callable): The image transformation function.
        output_dir (str): The directory to save the OCR results.
        batch_size (int): The maximum number of images per batch.
        device (str): The device to run inference on.
        batch_timeout (int): The timeout for processing a partial batch.
        processed (set): The set of already processed filenames to update.
        stop_event (threading.Event): A signal to gracefully shut down.
        lock (threading.Lock): The lock for thread-safe access to processed set.
    """
    
    pending = []
    last_process = time.time()
    waiting_message_printed = False

    logger.info("ParSeq processor thread started.")
    
    while not stop_event.is_set() or not queue.empty() or pending:
        try:
            while len(pending) < batch_size and not stop_event.is_set():
                path = queue.get(timeout=batch_timeout)
                pending.append(path)
        except Empty:
            pass

        now = time.time()
        
        if pending and (len(pending) >= batch_size or (now - last_process) >= batch_timeout or stop_event.is_set()):
            logger.info("Processing batch of %d images...", len(pending))
            waiting_message_printed = False
            results = recognize_batch(model, img_transform, pending, device)
            
            for path, (pred, conf) in results.items():
                base = path.stem
                out_path = Path(output_dir) / f"{base}.txt"
                try:
                    with open(out_path, 'w') as f:
                        f.write(f"{pred}\n{' '.join(f'{c:.4f}' for c in conf)}\n")
                    logger.info("Result for %s: %s", path.name, pred)
                    with lock:
                        processed.add(base)
                except Exception:
                    logger.error("An error occurred while saving OCR results for %s.", path.name, exc_info=True)
            pending.clear()
            last_process = now
        
        else:
            if not pending and not waiting_message_printed and (now - last_process) >= 10 and not stop_event.is_set():
                logger.info("Waiting for new images...")
                waiting_message_printed = True
                last_process = now
    
    logger.info("ParSeq processor thread finished its tasks and is exiting gracefully.")

def main():
    
    args = get_args()

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    setup_logging('lpr_service.log')

    try:
        logger.info("Loading ParSeq model from %s...", args.checkpoint)
        model = ModelClass.load_from_checkpoint(args.checkpoint).eval().to(args.device)
        img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    except Exception:
        logger.error("Failed to load the ParSeq model. Please check the checkpoint path and file integrity.", exc_info=True)
        return

    processed = get_already_processed(args.output_dir)

    batch_queue = Queue()
    stop_event = threading.Event()
    lock = threading.Lock()
    
    process_existing_files(args.input_dir, batch_queue, processed)

    processor_thread = threading.Thread(
        target=batch_processor,
        args=(batch_queue, model, img_transform, args.output_dir, args.batch_size, args.device, args.batch_timeout, processed, stop_event, lock),
        daemon=True
    )
    processor_thread.start()

    handler = LPRHandler(batch_queue, processed, lock)
    observer = Observer()
    observer.schedule(handler, args.input_dir, recursive=True)
    observer.start()

    logger.info("LPR service has started. Watching '%s' and its subdirectories for new images...", args.input_dir)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. LPR service stop requested by user.")
        observer.stop()
        stop_event.set()
        logger.info("Waiting for the LPR processor thread to finish its work...")
        observer.join()
        processor_thread.join()
    finally:
        logger.info("LPR service has shut down completely.")
        
if __name__ == "__main__":
    start = time.time()
    main()
    logger.info("LPR service runtime: %.2f seconds", time.time() - start)