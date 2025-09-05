# License Plate Detection Service
#
# This script processes a directory of images to automatically detect and crop license plates.
# It is designed as the initial stage of an Automated License Plate Recognition pipeline.

import os
import cv2
import time
import argparse
import threading
import logging
import torch
import json
from queue import Queue, Empty
from pathlib import Path

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
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

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
    parser.add_argument('--input_dir',      type=str, required=True,    help='Directory with input images')
    parser.add_argument('--output_dir',     type=str, required=True,    help='Directory to save cropped license plates')
    parser.add_argument('--model_path',     type=str, required=True,    help='Path to YOLO model for LPD')
    parser.add_argument('--imgsz',          type=int, default=640,      help='Image size for LPD inference')
    parser.add_argument('--batch_size',     type=int, default=16,       help='Batch size for LPD inference')
    parser.add_argument('--batch_timeout',  type=int, default=5,        help='Timeout in seconds to process a partial batch')
    parser.add_argument('--device',         type=str, default='cuda:0', help='Device to use for inference (e.g., "cuda:0", "cpu", "mps")')
    parser.add_argument('--crop',           action='store_true',       help='If set, save cropped license plate images to the output directory')
    parser.add_argument('--preds_file',     type=str, default=None,     help='Path to file where YOLO predictions will be saved (jsonl). If not provided, defaults to <output_dir>/yolo_predictions.jsonl')
    return parser.parse_args()


def get_already_processed(output_dir):
    """
    Identifies images already processed by checking for existing cropped license plate files.
    The check is based on a specific naming convention: `_lp` suffix in the filename.

    Args:
        output_dir (str): The directory where cropped license plates are saved.

    Returns:
        set: A set of base filenames (without extension or the `_lp` suffix) that have been processed.
    """
    processed = set()
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.png') and '_lp' in file:
                base = file.split('_lp')[0]
                processed.add(base)
    logger.info("Found %d already processed images in the output directory.", len(processed))
    return processed


def _write_predictions_to_file(entry, preds_file, lock):
    """Write a single JSON line (entry) into preds_file with thread-safety."""
    try:
        with lock:
            with open(preds_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception:
        logger.error("Failed to write predictions for %s to %s", entry.get('image'), preds_file, exc_info=True)


def save_lp_crops(results_batch, image_paths, output_dir, processed_set, crop_flag, preds_file, preds_lock, processed_lock):
    """
    Saves the cropped license plates from a batch of images based on YOLO detection results.

    Also always saves YOLO predicted values (one JSON per input image) into a separate file.

    The predictions file will use the same filename convention as the saved crops:
    - For each detected box a separate JSON line will be written with `image` set to
      `<base_name>_lp{n}.png` (matching the crop filename).
    - If no boxes are detected, a single entry with `image` set to `<base_name>_lp0.png` is
      written and `predictions` will be an empty list.

    Args:
        results_batch (list): A list of YOLO results objects, one for each image in the batch.
        image_paths (list): A list of the corresponding paths for the images in the batch.
        output_dir (str): The directory to save the cropped images.
        processed_set (set): A set to track processed filenames.
        crop_flag (bool): When True, crops will be written to disk. Otherwise crops are skipped.
        preds_file (str): Path to the file which will receive YOLO predictions in jsonl format.
        preds_lock (threading.Lock): Lock to ensure predictions file writes are thread-safe.
        processed_lock (threading.Lock): Lock to ensure processed_set is thread-safe.
    """
    for result, img_path in zip(results_batch, image_paths):
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        logger.debug("Attempting to read and process image: %s", img_name)
        image = cv2.imread(img_path)

        if image is None:
            logger.error("Error: Image file %s could not be read. The file may be corrupt or unsupported. Skipping.", img_name)
            # Still mark and write an empty prediction entry using crop-style naming
            entry_image = f"{base_name}_lp0.png"
            entry = {'image': entry_image, 'predictions': []}
            _write_predictions_to_file(entry, preds_file, preds_lock)
            with processed_lock:
                processed_set.add(base_name)
            continue

        # Prepare predictions list to always save
        preds = []
        try:
            # Attempt fast vector extraction
            if hasattr(result.boxes, 'xyxy') and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else [None] * len(xyxy)
                clss = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, 'cls') else [None] * len(xyxy)

                for i, bbox in enumerate(xyxy):
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    conf = float(confs[i]) if len(confs) > i else None
                    cls = int(clss[i]) if len(clss) > i else None
                    preds.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'class': cls})
            else:
                # Fallback: iterate boxes
                for box in result.boxes:
                    try:
                        x_center, y_center, w, h = box.xywh.squeeze().cpu().numpy()
                        x1 = float(max(x_center - w / 2, 0))
                        y1 = float(max(y_center - h / 2, 0))
                        x2 = float(min(x_center + w / 2, image.shape[1]))
                        y2 = float(min(y_center + h / 2, image.shape[0]))
                        conf = float(box.conf.cpu().numpy()) if hasattr(box, 'conf') else None
                        cls = int(box.cls.cpu().numpy()) if hasattr(box, 'cls') else None
                        preds.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'class': cls})
                    except Exception:
                        logger.debug("Failed to extract bbox from a Box object; skipping this box.", exc_info=True)
        except Exception:
            logger.error("Unexpected error while extracting predictions for image %s", img_name, exc_info=True)

        # If there are no predictions, write a single entry using the lp0 convention
        if len(preds) == 0:
            entry_image = f"{base_name}_lp0.png"
            entry = {'image': entry_image, 'predictions': []}
            _write_predictions_to_file(entry, preds_file, preds_lock)

            if not crop_flag:
                logger.info("No license plate detected in image: %s", img_name)
            else:
                logger.info("No license plate detected in image: %s", img_name)
            
            with processed_lock:
                processed_set.add(base_name)
            continue

        # If there are predictions, write one JSON line per predicted box using the crop naming
        for idx, p in enumerate(preds):
            entry_image = f"{base_name}_lp{idx+1}.png"
            entry = {'image': entry_image, 'predictions': [p]}
            _write_predictions_to_file(entry, preds_file, preds_lock)

        # If crop flag is False, skip actual image cropping but still wrote prediction entries above
        if not crop_flag:
            logger.info("YOLO detected %d boxes in image %s but crop flag is disabled; skipping crop saving.", len(preds), img_name)
            with processed_lock:
                processed_set.add(base_name)
            continue

        # Cropping behaviour (only if crop_flag True)
        if len(preds) > 1:
            logger.warning("Multiple (%d) license plates detected in image: %s. Processing all.", len(preds), img_name)

        for idx, p in enumerate(preds):
            try:
                x1, y1, x2, y2 = [int(round(v)) for v in p['bbox']]
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, image.shape[1])
                y2 = min(y2, image.shape[0])

                logger.debug("Cropping license plate #%d from %s with coordinates: [%d, %d, %d, %d]", idx + 1, img_name, x1, y1, x2, y2)

                crop = image[y1:y2, x1:x2]
                out_name = f"{base_name}_lp{idx+1}.png"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, crop)
                logger.info("Successfully saved license plate crop: %s", out_path)

            except Exception:
                logger.error("An error occurred while processing and saving a crop from image %s.", img_name, exc_info=True)
        
        with processed_lock:
            processed_set.add(base_name)


def process_existing_files(input_dir, queue, processed):
    """
    Scans the input directory and all its subdirectories to queue any existing unprocessed images.
    
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
        for fpath_obj in input_path.rglob('*'):

            if fpath_obj.is_file() and fpath_obj.suffix.lower() in valid_extensions:
                fpath = str(fpath_obj)
                stem = fpath_obj.stem
                all_files += 1
                
                if stem in processed:
                    skipped += 1
                    continue
                
                queue.put(fpath)
                logger.info("Queued existing file for processing: %s", fpath)

    logger.info("Initial scan complete. Total images found: %d, Skipped: %d, Queued: %d", all_files, skipped, all_files - skipped)


class LPDHandler(FileSystemEventHandler):
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
        file_path = event.src_path
        file_name = os.path.basename(file_path)

        if event.is_directory or os.path.splitext(file_name)[1].lower() not in valid_extensions:
            logger.debug("Ignoring new item (not a valid image file): %s", file_name)
            return
        
        stem = os.path.splitext(file_name)[0]
        with self.lock:
            if stem in self.processed:
                logger.info("Skipping new file: %s (already processed)", file_name)
                return
        
        logger.info("New image file detected: %s. Adding to processing queue.", file_name)
        self.queue.put(file_path)


def batch_processor(queue, model, args, processed_set, stop_event, preds_file, preds_lock, processed_lock):
    """
    Worker function that runs in a separate thread.
    Pulls image paths from the queue and processes them in batches.
    
    Args:
        queue (Queue): The thread-safe queue for file paths.
        model (YOLO): The loaded YOLO model.
        args (Namespace): Parsed command-line arguments.
        processed_set (set): The set of processed filenames.
        stop_event (threading.Event): A signal to gracefully shut down.
        preds_file (str): Path to save predictions (jsonl).
        preds_lock (threading.Lock): Lock to guard preds file writes.
        processed_lock (threading.Lock): Lock to guard processed_set writes.
    """
    pending_paths = []
    last_process_time = time.time()
    waiting_message_printed = False
    
    logger.info("LPD processor thread started on device: %s", args.device)

    while not stop_event.is_set() or not queue.empty() or pending_paths:
        try:
            while len(pending_paths) < args.batch_size and not stop_event.is_set():
                timeout = args.batch_timeout - (time.time() - last_process_time)
                path = queue.get(timeout=max(0.1, timeout))
                pending_paths.append(path)
                logger.debug("Added file to pending batch: %s", os.path.basename(path))
        except Empty:
            pass

        now = time.time()
        
        if pending_paths and (len(pending_paths) >= args.batch_size or (now - last_process_time) >= args.batch_timeout or stop_event.is_set()):
            logger.info("Processing batch of %d images.", len(pending_paths))
            
            # Reset the flag as we're no longer in a waiting state
            waiting_message_printed = False
            
            batch_imgs = []
            valid_paths = []
            
            for p in pending_paths:
                try:
                    img = cv2.imread(p)
                    if img is not None:
                        batch_imgs.append(img)
                        valid_paths.append(p)
                        logger.debug("Successfully read image: %s", os.path.basename(p))
                    else:
                        logger.warning("OpenCV failed to read file. Skipping: %s", os.path.basename(p))
                
                except Exception:
                    logger.error("An error occurred while reading file: %s", os.path.basename(p), exc_info=True)
            
            if batch_imgs:
                try:
                    logger.debug("Starting YOLO inference on batch with %d images.", len(batch_imgs))
                    results = model(batch_imgs, imgsz=args.imgsz, verbose=False, device=args.device)
                    save_lp_crops(results, valid_paths, args.output_dir, processed_set, args.crop, preds_file, preds_lock, processed_lock)
                    logger.info("Batch inference complete. Processed %d images and saved %d crops.", len(valid_paths), sum(len(r.boxes) for r in results))
                
                except Exception:
                    logger.error("An error occurred during model inference.", exc_info=True)
            
            pending_paths.clear()
            last_process_time = now
        
        else:
            if not pending_paths and not waiting_message_printed and (now - last_process_time) >= 10 and not stop_event.is_set():
                logger.info("Waiting for new images...")
                waiting_message_printed = True
                last_process_time = now
    
    logger.info("LPD processor thread finished its tasks and is exiting gracefully.")


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine predictions file path (must be different from the log file)
    preds_file = args.preds_file if args.preds_file else os.path.join(args.output_dir, 'yolo_predictions.jsonl')
    preds_dir = os.path.dirname(preds_file)
    if preds_dir and not os.path.exists(preds_dir):
        os.makedirs(preds_dir, exist_ok=True)

    # Logging kept separate and unchanged
    setup_logging('lpd_service.log')

    processed_basenames = get_already_processed(args.output_dir)
    
    logger.info("Loading YOLO model from %s...", args.model_path)
    device = torch.device(args.device)
    model = YOLO(args.model_path)
    model.to(device)
    
    processing_queue = Queue()
    stop_event = threading.Event()

    # Create a lock for predictions file writes
    preds_lock = threading.Lock()
    processed_lock = threading.Lock()

    # Ensure preds file exists (touch)
    try:
        open(preds_file, 'a', encoding='utf-8').close()
    except Exception:
        logger.error("Unable to create or touch predictions file: %s", preds_file, exc_info=True)

    process_existing_files(args.input_dir, processing_queue, processed_basenames)
    
    processor_thread = threading.Thread(
        target=batch_processor,
        args=(processing_queue, model, args, processed_basenames, stop_event, preds_file, preds_lock, processed_lock),
        daemon=True
    )
    processor_thread.start()
    
    event_handler = LPDHandler(processing_queue, processed_basenames, processed_lock)
    observer = Observer()
    observer.schedule(event_handler, args.input_dir, recursive=True)
    observer.start()

    logger.info("LPD service has started. Watching '%s' and its subdirectories for new images...", args.input_dir)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. LPD service stop requested by user.")
        
        observer.stop()
        stop_event.set()
        
        logger.info("Waiting for the LPD processor thread to finish its work...")
        
        observer.join()
        processor_thread.join()
        
    finally:
        logger.info("LPD service has shut down completely.")


if __name__ == "__main__":
    start = time.time()
    main()
    logger.info("LPD service runtime: %.2f seconds", time.time() - start)