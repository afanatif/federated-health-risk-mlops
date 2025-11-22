import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from multiple clients using weighted average.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples from each client
    
    Returns:
        Aggregated metrics dictionary
    """
    # Calculate total number of examples
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    if total_examples == 0:
        return {}
    
    # Initialize aggregated metrics
    aggregated = {}
    
    # Get all metric keys from first client
    if metrics:
        metric_keys = metrics[0][1].keys()
        
        # Aggregate each metric
        for key in metric_keys:
            weighted_sum = sum(
                num_examples * m[key] 
                for num_examples, m in metrics 
                if key in m
            )
            aggregated[key] = weighted_sum / total_examples
    
    return aggregated


if __name__ == "__main__":
    # Use FedAvg strategy with metrics aggregation
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,                    # Use all available clients for training
        fraction_evaluate=1.0,               # Use all available clients for evaluation
        min_fit_clients=3,                   # Minimum clients for training
        min_evaluate_clients=3,              # Minimum clients for evaluation
        min_available_clients=3,             # Minimum clients that must connect
        fit_metrics_aggregation_fn=weighted_average,      # Aggregate training metrics
        evaluate_metrics_aggregation_fn=weighted_average, # Aggregate evaluation metrics
    )
    
    print("=" * 60)
    print("Starting Flower Federated Learning Server")
    print("=" * 60)
    print(f"Strategy: FedAvg")
    print(f"Rounds: 3")
    print(f"Min clients: 3")
    print(f"Metrics tracking: ENABLED (accuracy will be displayed)")
    print("=" * 60)
    
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
