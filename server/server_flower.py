import flwr as fl

if __name__ == "__main__":
    # Use FedAvg strategy and set the number of rounds
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,    # fraction of clients used for training
        fraction_eval=1.0,   # fraction of clients used for evaluation
        min_fit_clients=3,   # minimum clients to start training
        min_eval_clients=3,  # minimum clients to start evaluation
        min_available_clients=3,
        eval_fn=None,        # optional evaluation function
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),  # <- now passed via ServerConfig
        strategy=strategy
    )
