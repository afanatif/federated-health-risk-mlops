# server/server_flower.py
import flwr as fl
from flwr.server.server import ServerConfig  # correct import

if __name__ == "__main__":
    # Create a server config
    config = ServerConfig(num_rounds=3)

    # Start Flower server
    fl.server.start_server(
        server_address="localhost:8080",
        config=config
    )
