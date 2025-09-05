import os
import torch

class CheckpointManager:
    """
    Handles saving and loading of training checkpoints, tracking both the best
    primary metric and the best validation loss.
    """

    def __init__(self, save_dir=None, primary_metric='accuracy', mode='max'):
        """
        Initializes the manager.

        Args:
            save_dir (str): Directory where checkpoints will be saved.
            primary_metric (str): The key in the metrics dict to monitor for the 'best' model.
            mode (str): The mode for the primary metric ('min' or 'max').
        """
        
        self.save_dir = save_dir
        self.primary_metric = primary_metric
        self.mode = mode
        
        # Initialize trackers for both best models
        self.best_metric_val = float('inf') if mode == 'min' else float('-inf')
        self.best_val_loss = float('inf')
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def save(self, epoch, model, optimizer, scheduler, metrics, loss, gradnorm=None, gradnorm_optimizer=None):
        """
        Saves the complete training state and handles the two 'best' model checkpoints.
        Optionally, it also saves GradNorm components.
        
        Args:
            ...
            metrics (dict): Dictionary of metrics (e.g., accuracy, F1). Does not need to contain the loss.
            loss (float): The validation loss value for the current epoch.
        """

        if self.save_dir is None:
            raise ValueError("A 'save_dir' must be provided to the constructor to save checkpoints.")
        
        # Create the state dictionary for saving
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'metrics': {**metrics, 'loss': loss},
            'best_metric_val': self.best_metric_val,
            'best_val_loss': self.best_val_loss,
        }

        # Add GradNorm states (if provided)
        if gradnorm:
            state['gradnorm_state'] = gradnorm.state_dict()
        if gradnorm_optimizer:
            state['gradnorm_optimizer_state'] = gradnorm_optimizer.state_dict()
        
        # Always save the latest checkpoint for resumption
        torch.save(state, os.path.join(self.save_dir, 'last.pt'))
        
        # 'best_metric' checkpoint
        if self.primary_metric in metrics:
            current_metric_val = metrics[self.primary_metric]
            
            is_best_metric = (self.mode == 'min' and current_metric_val < self.best_metric_val) or \
                             (self.mode == 'max' and current_metric_val > self.best_metric_val)

            if is_best_metric:
                self.best_metric_val = current_metric_val
                torch.save(state, os.path.join(self.save_dir, f'best_{self.primary_metric}.pt'))
                print(f"New best model for metric '{self.primary_metric}' found! Checkpoint saved.")
        
        # 'best_loss' checkpoint
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            torch.save(state, os.path.join(self.save_dir, 'best_loss.pt'))
            print(f"New best model for loss found! Checkpoint saved.")

    def load(self, path, device, model, optimizer=None, scheduler=None, gradnorm=None, gradnorm_optimizer=None):
        """
        Loads the training state from a checkpoint file.
        Optionally, it also saves GradNorm components.
        """
        
        if not os.path.exists(path):
            raise ValueError(f"Checkpoint not found at '{path}'. Starting training from scratch.")

        checkpoint = torch.load(path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state'])
        
        if optimizer and 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if scheduler and 'scheduler_state' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state'])

        if gradnorm and 'gradnorm_state' in checkpoint:
            gradnorm.load_state_dict(checkpoint['gradnorm_state'])
        
        if gradnorm_optimizer and 'gradnorm_optimizer_state' in checkpoint:
            gradnorm_optimizer.load_state_dict(checkpoint['gradnorm_optimizer_state'])

            
        start_epoch = checkpoint['epoch'] + 1
        metrics = checkpoint.get('metrics', {})
        
        # Restore the best known values to continue tracking correctly
        self.best_metric_val = checkpoint.get('best_metric_val', self.best_metric_val)
        self.best_val_loss = checkpoint.get('best_val_loss', self.best_val_loss)
        
        print(f"Resuming training from epoch {start_epoch}.")
        
        return start_epoch, metrics

    @staticmethod
    def _load_legacy(path, model, device):
        """
        Loads a legacy model checkpoint from a specified path.
        """
        if not os.path.exists(path):
            raise ValueError(f"Checkpoint not found at '{path}'")

        model.load_state_dict(torch.load(path, map_location=device))