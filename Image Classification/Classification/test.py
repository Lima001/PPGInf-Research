import argparse
import os
import torch
import yaml
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from ultralytics.data.augment import classify_transforms

from lib.dataset import MultiTaskDataset, multitask_collate_fn
from lib.model import build_model
from lib.device import select_device
from lib.metrics import compute_metrics

# --- Helper Functions ---

def load_config(path='config.yaml'):
    """Loads a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Unified testing script for multi-task models.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to YAML config file")
    parser.add_argument('--idx', type=int, required=True, help="The fold or split index for testing.")
    parser.add_argument('--report-summary', action='store_true', help='Generate a summary report with metrics.')
    parser.add_argument('--report-per-image', action='store_true', help='Print prediction, ground truth, and confidence for each image.')
    return parser.parse_args()

def print_metrics_report(avg_metrics, per_task_metrics):
    """Helper function to print the final metrics report."""
    print("\n--- Summary Report: Evaluation Metrics ---")
    for task_name, metrics in per_task_metrics.items():
        print(f"[{task_name}] Micro Acc: {metrics['micro_acc']:.4f} | Macro Acc: {metrics['macro_acc']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
    
    print(f"\nOverall Micro Acc : {avg_metrics['micro_acc']:.4f}")
    print(f"Overall Macro Acc : {avg_metrics['macro_acc']:.4f}")
    print(f"Overall Macro F1  : {avg_metrics['macro_f1']:.4f}")

# --- Main Execution ---

def main():
    args = parse_args()
    cfg = load_config(args.config)

    if not args.report_summary and not args.report_per_image:
        print("Error: No report type specified. Please use --report-summary and/or --report-per-image.")
        return
    
    # Basic setup
    data_cfg = cfg['data']
    testing_cfg = cfg['testing']
    device = select_device(testing_cfg.get('device', 'cuda'))
    task_classes = data_cfg['task_classes']
    task_names = list(task_classes.keys())

    # Dataset and dataLoader 
    test_transform = classify_transforms()
    
    test_ds = MultiTaskDataset(
        os.path.join(data_cfg['csv_path'], str(args.idx), "test.txt"),
        data_cfg['img_dir'],
        transform=test_transform,
        task_names=task_names
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=testing_cfg['batch_size'],
        shuffle=False,
        collate_fn=multitask_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Model loading based on strategy (multitask or ensemble)
    models = {}
    
    if testing_cfg['strategy'] == 'multitask':
        print("Strategy: Loading single multi-task model...")
        
        model = build_model(
            model_name=testing_cfg['model']['name'],
            task_classes=task_classes,
            device=device,
            freeze=False
        )
        
        weights_path = os.path.join(testing_cfg['model']['weights_dir'], str(args.idx), 'best_acc.pt')
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        
        models['multitask'] = model

    elif testing_cfg['strategy'] == 'ensemble':
        print("Strategy: Loading ensemble of single-task models...")
        
        for task in task_names:
            task_config = testing_cfg['ensemble_models'][task]
            
            model = build_model(
                model_name=task_config['name'],
                task_classes={task: task_classes[task]},
                device=device,
                freeze=False
            )
            
            weights_path = os.path.join(task_config['weights_dir'], str(args.idx), 'best_acc.pt')
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.eval()
            
            models[task] = model
    
    else:
        raise ValueError(f"Unknown testing strategy: {testing_cfg['strategy']}")

    # Initialize collectors (if needed) for the summary report
    if args.report_summary:
        y_true, y_pred = defaultdict(list), defaultdict(list)

    # Inference loop
    with torch.no_grad():
        for imgs, targets, fnames in tqdm(test_loader, desc="Testing"):

            imgs, targets = imgs.to(device), {k: v.to(device) for k, v in targets.items()}

            # Get model outputs
            outputs = {}
            if testing_cfg['strategy'] == 'multitask':
                outputs = models['multitask'](imgs)
            
            elif testing_cfg['strategy'] == 'ensemble':
                for task, model in models.items():
                    outputs.update(model(imgs))

            # Process outputs for each image in the batch
            for i in range(len(imgs)):
                pred_tuple, gt_tuple, conf_tuple = [], [], []
                
                for t in task_names:
                    logits = outputs[t][i]
                    probs = torch.softmax(logits, dim=0)
                    pred, conf = probs.argmax().item(), probs.max().item()
                    gt = targets[t][i].item()

                    pred_tuple.append(pred)
                    gt_tuple.append(gt)
                    conf_tuple.append(round(conf, 4))
                    
                    if args.report_summary:
                        y_true[t].append(gt)
                        y_pred[t].append(pred)

                if args.report_per_image:
                    fname_str = fnames[i]
                    pred_str = str(tuple(pred_tuple))
                    gt_str = str(tuple(gt_tuple))
                    conf_str = str(tuple(conf_tuple))
                    
                    print(f"{fname_str} | pred: {pred_str} | gt: {gt_str} | conf: {conf_str}")

    # Generate final reports (if requested)
    if args.report_summary:
        avg_metrics, per_task_metrics = compute_metrics(y_true, y_pred, task_classes)
        print_metrics_report(avg_metrics, per_task_metrics)

if __name__ == '__main__':
    main()