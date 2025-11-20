# server/server_flower.py
import flwr as fl

from flwr.server.server import ServerConfig

def main():
    # Start Flower server
    fl.server.start_server(
        server_address="localhost:8080",
        config=ServerConfig(num_rounds=3),
    )

if __name__ == "__main__":
    main()
