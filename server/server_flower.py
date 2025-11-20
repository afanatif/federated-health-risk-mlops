import flwr as fl

if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        num_rounds=3  # pass directly, NOT inside a dict
    )
