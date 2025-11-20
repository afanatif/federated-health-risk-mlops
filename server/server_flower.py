# server/server_flower.py
import flwr as fl
from flwr.server.server_config import ServerConfig  # correct import

if __name__ == "__main__":
    config = ServerConfig(num_rounds=3)
    fl.server.start_server(
        server_address="localhost:8080",
        config=config
    )
