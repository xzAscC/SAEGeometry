import jaxtyping
import torch
import numpy as np  
import random

def set_seed(seed: int) -> None:
    """Set the seed for reproducibility.

    Args:
        seed: The seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

def get_device() -> torch.device:
    """Get the device.

    Returns:
        The device.
    """
    # TODO: multi-gpu
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger:
    pass