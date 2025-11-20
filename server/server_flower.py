import flwr as fl

if __name__ == "__main__":
    # Use FedAvg strategy and set the number of rounds
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           # fraction of clients used for training
        fraction_evaluate=1.0,      # Changed from fraction_eval to fraction_evaluate
        min_fit_clients=3,          # minimum clients to start training
        min_evaluate_clients=3,     # Changed from min_eval_clients to min_evaluate_clients
        min_available_clients=3,
        evaluate_fn=None,           # Changed from eval_fn to evaluate_fn
    )
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
