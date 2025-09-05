import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau, StepLR

def build_scheduler(optimizer, policy, lr, epochs, **kwargs):
    """
    Builds a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        policy: The scheduler policy name (e.g., 'cosine', 'steplr').
        lr: The initial learning rate.
        epochs: The total number of training epochs.
        **kwargs: Extra scheduler-dependent arguments.

    Returns:
        _LRScheduler: The configured learning rate scheduler.
    """
    policy = policy.lower()
    
    if policy == "cosine":
        # Cosine schedule from `lr` down to a fraction of `lr`.
        lrf = kwargs.get('lrf', 0.01)
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * lrf)
        
    elif policy == "linear":
        # Linear schedule from `lr` down to a fraction of `lr`.
        lrf = kwargs.get('lrf', 0.01)
        lr_lambda = lambda epoch: 1.0 - (1.0 - lrf) * (epoch / (epochs - 1))
        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif policy == "steplr":
        # Decays the LR by a factor every `step_size` epochs.
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    elif policy == "reducelronplateau":
        # Reduces LR when a metric has stopped improving.
        patience = kwargs.get('patience', 10)
        factor = kwargs.get('factor', 0.1)
        return ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

    else:
        raise ValueError(f"Unsupported scheduler policy: '{policy}'")