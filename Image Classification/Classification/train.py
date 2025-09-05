import argparse
import os
import torch
import yaml

from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.checkpoint import CheckpointManager
from lib.dataset import build_dataloaders
from lib.device import select_device
from lib.early_stopping import EarlyStopping
from lib.loss import build_loss_fn_scalable
from lib.metrics import compute_metrics
from lib.model import build_model
from lib.optimizer import build_optimizer
from lib.scheduler import build_scheduler

# --- Configuration Loading ---

def load_config(path):
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-task image classification training script.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--idx", type=int, required=True, help="The fold or split index for training.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint.")
    return parser.parse_args()


# --- Core Training/Validation Logic ---

def run_epoch(model, loader, device, criterion, optimizer=None, task_weights=None, is_train=True):
    """Runs a single training or validation epoch."""
    
    model.train(is_train)
    
    total_loss, samples = 0.0, 0
    y_true, y_pred = defaultdict(list), defaultdict(list)
    
    desc = "Train" if is_train else "Validate"
    desc = desc.ljust(8)
    pbar = tqdm(loader, desc=desc, ncols=120)

    with torch.set_grad_enabled(is_train):
        
        for imgs, targets, _ in pbar:
            imgs = imgs.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            if is_train:
                optimizer.zero_grad()

            outputs = model(imgs)
            classification_loss = torch.tensor(0.0, device=device)
            
            for task, logits in outputs.items():
                loss = criterion(logits, targets[task])
                weight = task_weights.get(task, 1.0)
                classification_loss += loss * weight
                preds = logits.argmax(dim=1)
                y_true[task].extend(targets[task].cpu().tolist())
                y_pred[task].extend(preds.cpu().tolist())
            
            batch_loss = classification_loss

            if is_train:
                batch_loss.backward()
                optimizer.step()

            batch_size = imgs.size(0)
            total_loss += batch_loss.item() * batch_size
            samples += batch_size
            pbar.set_postfix(loss=f"{total_loss / samples:.4f}")

    return (total_loss/samples), y_true, y_pred

# --- Main Script ---

def main():
    """Main training and validation loop."""
    args = parse_args()
    cfg = load_config(args.config)

    # Deconstruct config for clarity
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    optim_cfg = cfg['optimizer']
    loss_cfg = cfg['loss']
    training_cfg = cfg['training']
    primary_metric = training_cfg['primary_metric']

    # Basic setup
    device = select_device(training_cfg['device'])
    save_dir = os.path.join(training_cfg['save_dir'], str(args.idx))

    # Build components from libs
    task_classes = data_cfg['task_classes']
    
    train_loader, val_loader = build_dataloaders(
        task_classes=task_classes, csv_path=data_cfg['csv_path'], img_dir=data_cfg['img_dir'],
        batch_size=training_cfg['batch_size'], num_workers=training_cfg['num_workers'], fold_idx=args.idx
    )
    
    model = build_model(
        model_name=model_cfg['name'], task_classes=task_classes,
        freeze=model_cfg['freeze_backbone'], device=device
    )
    
    optimizer = build_optimizer(
        model=model, optimizer_name=optim_cfg['name'], lr=optim_cfg['lr'],
        weight_decay=optim_cfg['weight_decay'], **optim_cfg.get('kwargs', {})
    )
    
    scheduler = build_scheduler(
        optimizer=optimizer, policy=training_cfg['scheduler']['policy'], lr=optim_cfg['lr'],
        epochs=training_cfg['epochs'], **training_cfg['scheduler'].get('kwargs', {})
    )
    
    criterion = build_loss_fn_scalable(loss_type=loss_cfg['name'], **loss_cfg.get('kwargs', {}))

    early_stopper = EarlyStopping(patience=training_cfg['patience'], mode="min")

    # Checkpoint and resumption
    checkpoint_manager = CheckpointManager(save_dir=save_dir, primary_metric=primary_metric, mode='max')
    start_epoch = 1
    
    # Resume only if --resume flag is set
    if args.resume:
        resume_path = os.path.join(save_dir, 'last.pt')
        start_epoch, _ = checkpoint_manager.load(resume_path, device, model, optimizer, scheduler, early_stopper=early_stopper)
        early_stopper.best_score = checkpoint_manager.best_val_loss

    # --- Training Loop ---
    task_weights = loss_cfg['task_weights']
    epochs = training_cfg['epochs']
    
    # Initialize trackers for the final report
    best_metric_tracker = {"epoch": 0, "value": float('-inf')}
    best_loss_tracker = {"epoch": 0, "value": float('inf')}

    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs} --- LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Train and Validate
        train_loss, y_true_train, y_pred_train = run_epoch(
            model, train_loader, device, criterion, optimizer, task_weights, is_train=True
        )
        
        val_loss, y_true_val, y_pred_val = run_epoch(
            model, val_loader, device, criterion, None, task_weights, is_train=False
        )

        # LR Scheduler Step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Metrics and logging
        avg_train_metrics, _ =  compute_metrics(y_true_train, y_pred_train, task_classes, task_weights)
        avg_val_metrics, per_task_val_metrics = compute_metrics(y_true_val, y_pred_val, task_classes, task_weights)

        # Print the main summary line with AVERAGED metrics
        print(f"-- Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"-- Avg Train Metrics -> Micro-Acc: {avg_train_metrics['micro_acc']:.4f}, Macro-Acc: {avg_train_metrics['macro_acc']:.4f}, Macro-F1: {avg_train_metrics['macro_f1']:.4f}")
        print(f"-- Avg Val   Metrics -> Micro-Acc: {avg_val_metrics['micro_acc']:.4f}, Macro-Acc: {avg_val_metrics['macro_acc']:.4f}, Macro-F1: {avg_val_metrics['macro_f1']:.4f}")

        # 3. Print the PER-TASK validation metrics in a separate, organized section
        print("-- Per-Task Validation Metrics:")
        for task_name, metrics in per_task_val_metrics.items():
            task_str = f"\t'{task_name}': "
            metrics_str = f"Micro-Acc: {metrics['micro_acc']:.4f}, Macro-Acc: {metrics['macro_acc']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}"
            print(task_str + metrics_str)

        # Update best value trackers
        current_metric = avg_val_metrics[primary_metric]
        
        if current_metric > best_metric_tracker["value"]:
            best_metric_tracker["value"] = current_metric
            best_metric_tracker["epoch"] = epoch
        
        if val_loss < best_loss_tracker["value"]:
            best_loss_tracker["value"] = val_loss
            best_loss_tracker["epoch"] = epoch

        early_stopper.step(val_loss)

        # Save Checkpoint
        checkpoint_manager.save(epoch, model, optimizer, scheduler, avg_val_metrics, val_loss, early_stopper=early_stopper)

        # Early stopping check
        if early_stopper.should_stop:
            print(f"\nEarly stopping triggered after {early_stopper.patience} epochs with no improvement.")
            break
        
        print()

    print("Training complete.")

    # Final report (reduced)
    print(f"Best Validation {primary_metric.capitalize()}: {best_metric_tracker['value']:.4f} (achieved at Epoch {best_metric_tracker['epoch']})")
    print(f"Best Validation Loss: {best_loss_tracker['value']:.4f} (achieved at Epoch {best_loss_tracker['epoch']})")
    print(f"Checkpoints saved in: {save_dir}")

if __name__ == "__main__":
    main()