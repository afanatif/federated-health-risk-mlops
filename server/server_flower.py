# server/server_flower.py
import flwr as fl

if __name__ == "__main__":
    # Simple Flower server with 3 rounds for demo
    fl.server.start_server(config={"num_rounds": 3})
