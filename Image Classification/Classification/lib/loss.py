import torch
import torch.nn as nn
import torch.nn.functional as F

class GradNorm(nn.Module):
    """
    Implements the GradNorm algorithm for balancing multi-task losses.
    It learns weights for each task loss to balance gradient magnitudes.
    """
    def __init__(self, task_names, alpha=1.5, target_layer=None):
        super().__init__()
        
        if target_layer is None:
            raise ValueError("GradNorm requires a target_layer to compute gradients.")
        
        self.task_names = task_names
        self.alpha = alpha
        self.T = len(task_names)
        self.target_layer = target_layer
        
        # Trainable weights for each task
        self.weights = nn.Parameter(torch.ones(self.T))
        self.initial_losses = None

    def normalize_weights(self):
        """Ensure weights sum to T for stability."""
        with torch.no_grad():
            self.weights.data = (self.weights / self.weights.sum() * self.T)

    def compute_gradnorm_loss(self, task_losses):
        """Calculates the GradNorm loss for updating the task weights."""
        if self.initial_losses is None:
            self.initial_losses = torch.stack([l.detach() for l in task_losses])

        # Compute gradient norms for each task w.r.t. the shared layer
        grad_norms = []
        for i in range(self.T):
            # We need to retain the graph to backpropagate the GradNorm loss
            grad = torch.autograd.grad(self.weights[i] * task_losses[i], self.target_layer.parameters(), retain_graph=True, create_graph=True)[0]
            grad_norms.append(torch.norm(grad))
        
        grad_norms = torch.stack(grad_norms)

        # Compute the GradNorm loss
        loss_ratios = torch.stack([l / (l0 + 1e-8) for l, l0 in zip(task_losses, self.initial_losses)])
        rt = loss_ratios / loss_ratios.mean()
        avg_grad_norm = grad_norms.mean().detach()
        target_grad_norms = (avg_grad_norm * rt ** self.alpha).detach()
        
        gradnorm_loss = torch.abs(grad_norms - target_grad_norms).sum()
        return gradnorm_loss


def build_loss_fn_scalable(loss_type, **kwargs):
    """
    A scalable factory to build a loss function using its specific parameters.

    Args:
        loss_type: The type of loss function to build.
        **kwargs: A dictionary of parameters for the specified loss function.

    Returns:
        A callable loss function.
    """
   
    if loss_type == 'cross_entropy':
        label_smoothing = kwargs.get('label_smoothing', 0.0)
        return lambda logits, true: F.cross_entropy(logits, true, label_smoothing=label_smoothing, reduction="mean")
    
    else:
        raise ValueError(f"Unknown loss type '{loss_type}' specified.")


def kl_divergence_consistency(logits, parent_labels, parent_to_child_mask):
    """
    Calculates KL divergence loss to enforce hierarchical consistency.

    This loss encourages the model's predicted distribution over child classes
    to align with a uniform distribution over the valid children for a given
    ground-truth parent class.

    Args:
        logits: Raw logits for child classes
        parent_labels: Ground-truth parent class indices
        parent_to_child_mask: A boolean or binary tensor of shape [num_parents, num_children], where mask[p, c] is True if child 'c' is valid for parent 'p'.

    Returns:
        Tensor: A scalar tensor representing the KL divergence loss.
    """
    
    with torch.no_grad():
        # Select the valid children for each parent label in the batch
        expected_probs = parent_to_child_mask[parent_labels].float()
        
        # Normalize to create a uniform probability distribution over valid children
        # Add a small epsilon to prevent division by zero if a parent has no children
        expected_probs /= expected_probs.sum(dim=1, keepdim=True) + 1e-8

    # Convert model logits to log probabilities
    log_probs = F.log_softmax(logits, dim=1)
    
    # Calculate KL divergence
    # reduction='batchmean' averages the loss over the batch
    return F.kl_div(log_probs, expected_probs, reduction='batchmean')