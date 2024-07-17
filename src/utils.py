import random

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Set the random seed for reproducibility.
    
    Args:
        seed (int): Random seed value.
    Returns:
        None
    """
    random.seed(seed)  # Setting random seed for Python's random module
    np.random.seed(seed)  # Setting random seed for numpy
    torch.manual_seed(seed)  # Setting random seed for PyTorch
