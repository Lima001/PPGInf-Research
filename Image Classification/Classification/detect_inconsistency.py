import argparse
import os
import yaml
import torch
from collections import defaultdict
from tqdm import tqdm
from ultralytics.data.augment import classify_transforms

from lib.dataset import MultiTaskDataset, multitask_collate_fn
from lib.device import select_device
from lib.model import build_model
from lib.checkpoint import CheckpointManager
from lib.utils import load_vehicle_hierarchy_masks

def load_config(path):
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Detect inconsistent predictions using a class hierarchy.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the model and data configuration file.")
    parser.add_argument('--idx', type=int, required=True, help="The fold or split index for the test set.")
    parser.add_argument('--extend', action='store_true', help="Enable detailed error breakdown by attribute combinations.")
    return parser.parse_args()

def main():
    """Main function to perform inference and detect inconsistencies."""
    args = parse_args()
    cfg = load_config(args.config)
    
    # Deconstruct config
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    inference_cfg = cfg['inference']
    hierarchy_cfg = cfg['hierarchy']

    # Basic setup
    device = select_device(inference_cfg.get('device', 'cuda'))
    task_classes = data_cfg['task_classes']
    task_names = sorted(list(task_classes.keys()))

    #  Load hierarchy masks
    type_to_make_mask, make_to_model_mask = load_vehicle_hierarchy_masks(
        json_path=hierarchy_cfg['json_path'],
        num_types=task_classes['vehicle_type'],
        num_makes=task_classes['vehicle_make'],
        num_models=task_classes['vehicle_model']
    )
    
    type_to_make_mask = type_to_make_mask.to(device)
    make_to_model_mask = make_to_model_mask.to(device)

    # Dataset and dataLoader
    transform = classify_transforms()
    
    test_ds = MultiTaskDataset(
        os.path.join(data_cfg['csv_path'], str(args.idx), "test.txt"),
        data_cfg['img_dir'],
        transform=transform,
        task_names=task_names
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=inference_cfg['batch_size'], shuffle=False,
        collate_fn=multitask_collate_fn, num_workers=inference_cfg.get('num_workers', 4)
    )

    # Model building
    model = build_model(
        model_name=model_cfg['name'], task_classes=task_classes,
        freeze=False, device=device
    )
    
    # Weights path is now read from the config file
    weights_path = inference_cfg['weights_path']
    manager = CheckpointManager()
    #manager.load(path=weights_path, device=device, model=model)
    # legacy code -> previous checkpoints only saved model weights (default torch.save call) 
    CheckpointManager._load_legacy(weights_path, model, device)
    model.eval()

    # Initialize counters
    total_errors = 0
    inconsistent_errors = 0
    total_samples = 0
    error_counts = defaultdict(int)

    # Inference and inconsistency check
    with torch.no_grad():
        for imgs, targets, _ in tqdm(test_loader, desc="Detecting Inconsistencies"):
            if targets is None: continue
            imgs, targets = imgs.to(device), {k: v.to(device) for k, v in targets.items()}
            outputs = model(imgs)

            for i in range(len(imgs)):
                preds = {t: outputs[t][i].argmax().item() for t in task_names}
                gts = {t: targets[t][i].item() for t in task_names}
                
                is_wrong = any(preds[t] != gts[t] for t in task_names)
                
                if is_wrong:
                    total_errors += 1
                    
                    pred_type = preds['vehicle_type']
                    pred_make = preds['vehicle_make']
                    pred_model = preds['vehicle_model']

                    type_make_ok = type_to_make_mask[pred_type, pred_make]
                    make_model_ok = make_to_model_mask[pred_make, pred_model]

                    if not (type_make_ok and make_model_ok):
                        inconsistent_errors += 1
                    
                    if args.extend:
                        error_keys = [t for t in task_names if preds[t] != gts[t]]
                        combo_key = '-'.join(sorted(error_keys))
                        error_counts[combo_key] += 1
            
            total_samples += len(imgs)
            
    # Final report
    print("--- Inconsistency Report ---")
    print(f"Total Samples Processed: {total_samples}")
    print(f"Total Samples with Errors: {total_errors}")
    print(f"Hierarchically Inconsistent Errors: {inconsistent_errors}")
    
    if total_errors > 0:
        inconsistent_ratio = inconsistent_errors / total_errors
        print(f"Of all errors, {inconsistent_ratio:.2%} were hierarchically inconsistent.")
    
    if args.extend and total_errors > 0:
        print("Detailed Error Breakdown by Task Combination")
        for combo, count in sorted(error_counts.items()):
            percent_of_total = count / total_errors * 100
            print(f"{combo.replace('vehicle_', ''):<25} | Count: {count:<5} ({percent_of_total:.2f}%)")

if __name__ == '__main__':
    main()