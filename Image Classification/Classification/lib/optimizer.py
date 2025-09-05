import torch
import torch.nn as nn

def build_optimizer(model, optimizer_name, lr, weight_decay, **kwargs):
    """
    Builds a simple optimizer (without parameter grouping).

    Args:
        model: The model to be optimized.
        optimizer_name: The name of the optimizer (e.g., 'adamw', 'sgd').
        lr: The learning rate.
        weight_decay: The weight decay value
        **kwargs: Extra optimizer-dependent arguments (e.g., momentum, betas).

    Returns:
        torch.optim.Optimizer: The configured optimizer.
    """
    
    # Get all parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Map optimizer names to their PyTorch classes
    optimizer_map = {
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'rmsprop': torch.optim.RMSprop,
    }
    
    optimizer_name = optimizer_name.lower()
    if optimizer_name not in optimizer_map:
        raise ValueError(f"Unsupported optimizer: '{optimizer_name}'. Available: {list(optimizer_map.keys())}")

    optimizer_class = optimizer_map[optimizer_name]

    # Instantiate the optimizer, passing common and extra arguments
    optimizer = optimizer_class(
        params,
        lr=lr,
        weight_decay=weight_decay,
        **kwargs
    )
    
    return optimizer