import argparse
import os
import torch
import torch.nn.functional as F
import yaml
from collections import defaultdict
from tqdm import tqdm

from lib.checkpoint import CheckpointManager
from lib.dataset import build_dataloaders
from lib.device import select_device
from lib.early_stopping import EarlyStopping
from lib.loss import GradNorm, kl_divergence_consistency, build_loss_fn_scalable
from lib.metrics import compute_metrics
from lib.model import build_model, get_gradnorm_target_layer
from lib.optimizer import build_optimizer
from lib.scheduler import build_scheduler
from lib.utils import disable_inplace_ops, load_vehicle_hierarchy_masks

# --- Helper Functions ---

def load_config(path='config.yaml'):
    """Loads a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="GradNorm Multi-task Training Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--idx", type=int, required=True, help="The fold or split index for training.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint.")
    return parser.parse_args()

# --- Core Logic Functions ---

def train_epoch_gradnorm(model, loader, device, criterion, optimizer, gradnorm, gradnorm_optimizer, masks, hierarchy_weight, task_names):
    """Runs a single training epoch using the GradNorm algorithm."""
    model.train()
    gradnorm.train()
    total_loss, samples = 0.0, 0
    y_true, y_pred = defaultdict(list), defaultdict(list)
    
    desc = "Train".ljust(8)
    pbar = tqdm(loader, desc=desc, ncols=120)

    for imgs, targets, _ in pbar:
        imgs, targets = imgs.to(device), {k: v.to(device) for k, v in targets.items()}
        outputs = model(imgs)
        
        task_losses = torch.stack([criterion(outputs[t], targets[t]) for t in task_names])

        # Update GradNorm weights
        gradnorm_loss = gradnorm.compute_gradnorm_loss(task_losses)
        gradnorm_optimizer.zero_grad()
        gradnorm_loss.backward(retain_graph=True)
        gradnorm_optimizer.step()
        gradnorm.normalize_weights()
        
        # Update model weights
        kl_loss_1 = kl_divergence_consistency(outputs['vehicle_make'], targets['vehicle_type'], masks['type_make'])
        kl_loss_2 = kl_divergence_consistency(outputs['vehicle_model'], targets['vehicle_make'], masks['make_model'])
        
        weighted_task_loss = torch.sum(gradnorm.weights * task_losses)
        batch_loss = weighted_task_loss + hierarchy_weight * (kl_loss_1 + kl_loss_2)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Collect metrics and update tracking variables
        for task, logits in outputs.items():
            y_pred[task].extend(logits.argmax(dim=1).cpu().tolist())
            y_true[task].extend(targets[task].cpu().tolist())

        batch_size = imgs.size(0)
        total_loss += batch_loss.item() * batch_size
        samples += batch_size
        pbar.set_postfix(loss=f"{total_loss / samples:.4f}", weights=[f"{w:.2f}" for w in gradnorm.weights.detach()])

    return (total_loss/samples), y_true, y_pred

def validate_epoch(model, loader, device, criterion, masks, hierarchy_weight, task_names):
    """Runs a single validation epoch."""
    model.eval()
    total_loss, samples = 0.0, 0
    y_true, y_pred = defaultdict(list), defaultdict(list)
    
    desc = "Validate".ljust(8)
    pbar = tqdm(loader, desc=desc, ncols=120)

    with torch.no_grad():
        for imgs, targets, _ in pbar:
            imgs, targets = imgs.to(device), {k: v.to(device) for k, v in targets.items()}
            outputs = model(imgs)
            
            task_losses = torch.stack([criterion(outputs[t], targets[t]) for t in task_names])
            kl1 = kl_divergence_consistency(outputs['vehicle_make'], targets['vehicle_type'], masks['type_make'])
            kl2 = kl_divergence_consistency(outputs['vehicle_model'], targets['vehicle_make'], masks['make_model'])
            
            batch_loss = task_losses.sum() + hierarchy_weight * (kl1 + kl2)

            for task, logits in outputs.items():
                y_pred[task].extend(logits.argmax(dim=1).cpu().tolist())
                y_true[task].extend(targets[task].cpu().tolist())
            
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
    hierarchy_cfg = cfg['hierarchy']
    gradnorm_cfg = cfg['gradnorm']
    primary_metric = training_cfg['primary_metric']

    # Basic setup
    device = select_device(training_cfg['device'])
    save_dir = os.path.join(training_cfg['save_dir'], str(args.idx))

    # Build components
    task_classes = data_cfg['task_classes']
    task_names = sorted(list(task_classes.keys()))
    
    train_loader, val_loader = build_dataloaders(
        task_classes=task_classes, csv_path=data_cfg['csv_path'], img_dir=data_cfg['img_dir'],
        batch_size=training_cfg['batch_size'], num_workers=training_cfg['num_workers'], fold_idx=args.idx
    )
    
    model = build_model(model_name=model_cfg['name'], task_classes=task_classes, freeze=model_cfg['freeze_backbone'], device=device)
    target_layer = get_gradnorm_target_layer(model)     # Get the specific layer for GradNorm
    disable_inplace_ops(model)
    
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

    # GradNorm and Hierarchy specific setup
    gradnorm = GradNorm(task_names, alpha=gradnorm_cfg['alpha'], target_layer=target_layer).to(device)
    gradnorm_optimizer = torch.optim.Adam([gradnorm.weights], lr=gradnorm_cfg['lr'])
    
    type_to_make, make_to_model = load_vehicle_hierarchy_masks(
        json_path=hierarchy_cfg['json_path'], num_types=task_classes['vehicle_type'],
        num_makes=task_classes['vehicle_make'], num_models=task_classes['vehicle_model']
    )
    
    masks = {'type_make': type_to_make.to(device), 'make_model': make_to_model.to(device)}

    # Checkpoint and Resumption
    checkpoint_manager = CheckpointManager(save_dir=save_dir, primary_metric=primary_metric, mode='max')
    start_epoch = 1
    
    if args.resume:
        resume_path = os.path.join(save_dir, 'last.pt')
        start_epoch, _ = checkpoint_manager.load(resume_path, device, model, optimizer, scheduler, early_stopper=early_stopper)
        early_stopper.best_score = checkpoint_manager.best_val_loss

    # Training Loop
    epochs = training_cfg['epochs']
    best_metric_tracker = {"epoch": 0, "value": float('-inf')}
    best_loss_tracker = {"epoch": 0, "value": float('inf')}

    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs} --- LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Train and Validate
        train_loss, y_true_train, y_pred_train = train_epoch_gradnorm(
            model, train_loader, device, criterion, optimizer, gradnorm, gradnorm_optimizer, 
            masks, hierarchy_cfg['weight'], task_names
        )

        val_loss, y_true_val, y_pred_val = validate_epoch(
            model, val_loader, device, criterion, masks, hierarchy_cfg['weight'], task_names
        )

        # LR Scheduler Step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Metrics and logging
        avg_train_metrics, _ = compute_metrics(y_true_train, y_pred_train, task_classes)
        avg_val_metrics, per_task_val_metrics = compute_metrics(y_true_val, y_pred_val, task_classes)

        print(f"-- Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"-- Avg Train Metrics -> Micro-Acc: {avg_train_metrics['micro_acc']:.4f}, Macro-Acc: {avg_train_metrics['macro_acc']:.4f}, Macro-F1: {avg_train_metrics['macro_f1']:.4f}")
        print(f"-- Avg Val   Metrics -> Micro-Acc: {avg_val_metrics['micro_acc']:.4f}, Macro-Acc: {avg_val_metrics['macro_acc']:.4f}, Macro-F1: {avg_val_metrics['macro_f1']:.4f}")
        print("-- Per-Task Validation Metrics:")
        for task_name, metrics in per_task_val_metrics.items():
            task_str = f"\t'{task_name}': "
            metrics_str = f"Micro-Acc: {metrics['micro_acc']:.4f}, Macro-Acc: {metrics['macro_acc']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}"
            print(task_str + metrics_str)

        # Update best value trackers and save checkpoint
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
    print(f"Best Validation {primary_metric.capitalize()}: {best_metric_tracker['value']:.4f} (at Epoch {best_metric_tracker['epoch']})")
    print(f"Best Validation Loss: {best_loss_tracker['value']:.4f} (at Epoch {best_loss_tracker['epoch']})")
    print(f"Checkpoints saved in: {save_dir}")

if __name__ == "__main__":
    main()