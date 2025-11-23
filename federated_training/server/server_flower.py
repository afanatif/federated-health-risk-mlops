# federated_training/server/server_flower.py
import flwr as fl
from federated_training.server.aggregation import fedavg
from federated_training.server.checkpoints import save_checkpoint, load_checkpoint

def start_server():
    # Use FedAvg by default
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,    # all clients participate
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=lambda rnd: {"round": rnd},
        evaluate_metrics_aggregation_fn=None,
        initial_parameters=None
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )
