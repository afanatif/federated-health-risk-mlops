# federated_training/server/aggregation.py
from typing import List
import numpy as np

# Type alias for clarity
Weights = List[np.ndarray]

def fedavg(weights_list: List[Weights]) -> Weights:
    """
    Standard FedAvg aggregation.
    """
    if not weights_list:
        raise ValueError("weights_list is empty")

    avg_weights = []
    for weights in zip(*weights_list):
        avg_weights.append(np.mean(weights, axis=0))
    return avg_weights
