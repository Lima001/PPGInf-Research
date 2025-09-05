import torch

def select_device(device_str=''):
    """
    Selects a torch.device based on a user's request, with clear validation.

    This function first checks for a 'cpu' request. If a GPU is requested,
    it verifies CUDA availability and validates the specific device index if one
    is provided. It will raise an error on any invalid or unavailable request.

    Args:
        device_str (str): The desired device ('cpu', 'cuda', or 'cuda:N'). If empty, defaults to 'cuda:0' if available.

    Returns:
        torch.device: The validated torch device.

    Raises:
        ValueError: If the device string is invalid or the index is out of range.
        RuntimeError: If a GPU is requested but CUDA is not available.
    """
    # Handle the explicit CPU request first.
    if device_str.lower() == 'cpu':
        print("Using CPU device.")
        return torch.device('cpu')

    # If not CPU, a GPU is intended. Verify CUDA is available.
    if not torch.cuda.is_available():
        raise RuntimeError("GPU request failed: CUDA is not available on this system.")

    # Handle auto-detect or generic 'cuda' request.
    if not device_str or device_str.lower() == 'cuda':
        print("CUDA available. Using default device: 'cuda:0'.")
        return torch.device('cuda:0')

    # Handle a specific GPU request like 'cuda:1'.
    if device_str.lower().startswith('cuda:'):
        try:
            index = int(device_str.split(':')[-1])
            num_gpus = torch.cuda.device_count()

            if index >= num_gpus:
                raise ValueError(f"Invalid device index: {index}. Only {num_gpus} GPUs available (indices 0 to {num_gpus - 1}).")
            
            print(f"Using specified device: 'cuda:{index}'.")
            return torch.device(f'cuda:{index}')
        except (ValueError, IndexError):
            raise ValueError(f"Invalid device string format: '{device_str}'. Expected 'cuda:N' where N is an integer.")

    # If the input is none of the above, it's an unrecognized format.
    raise ValueError(f"Unrecognized device string: '{device_str}'. Valid options are 'cpu', 'cuda', or 'cuda:N'.")