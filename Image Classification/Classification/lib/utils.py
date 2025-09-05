import json
import torch
import torchvision

def disable_inplace_ops(model):
    """
    Disables all in-place operations in a model.
    """
    for module in model.modules():
        if hasattr(module, 'inplace') and module.inplace is True:
            module.inplace = False
        
        # Specific check for some torchvision modules
        if isinstance(module, torchvision.ops.misc.ConvNormActivation) and hasattr(module, 'activation') and hasattr(module.activation, 'inplace'):
            module.activation.inplace = False


def load_vehicle_hierarchy_masks(json_path, num_types, num_makes, num_models):
    """
    Loads and creates boolean masks for a 3-level class hierarchy.

    The hierarchy is assumed to be: Type -> Make -> Model. The function reads a
    JSON file defining these relationships and creates two masks: one mapping
    valid makes for each type, and another mapping valid models for each make.

    Args:
        json_path: Path to the JSON file defining the hierarchy.
        num_types: The total number of 'type' classes.
        num_makes: The total number of 'make' classes.
        num_models: The total number of 'model' classes.

    Returns:
        A tuple containing:
        - type_make_mask: A boolean tensor of shape [num_types, num_makes].
        - make_model_mask: A boolean tensor of shape [num_makes, num_models].
    """
    
    with open(json_path, 'r') as f:
        hierarchy_data = json.load(f)

    # Initialize masks with all False
    type_make_mask = torch.zeros(num_types, num_makes, dtype=torch.bool)
    make_model_mask = torch.zeros(num_makes, num_models, dtype=torch.bool)

    # Populate masks based on the hierarchy data
    for type_entry in hierarchy_data:
        type_idx = int(type_entry['type'])
        for make_entry in type_entry['makes']:
            make_idx = int(make_entry['make'])
            
            # A make is valid for its parent type
            type_make_mask[type_idx, make_idx] = True
            
            for model_idx in make_entry['models']:
                # A model is valid for its parent make
                make_model_mask[make_idx, int(model_idx)] = True

    return type_make_mask, make_model_mask