# server/server_flower.py
from flwr.server import start_server, ServerConfig

if __name__ == "__main__":
    # Start Flower server with correct ServerConfig
    start_server(config=ServerConfig(num_rounds=3))
