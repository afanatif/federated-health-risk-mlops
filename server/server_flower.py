# server/server_flower.py
import flwr as fl

if __name__ == "__main__":
    # Start Flower server with 3 rounds (adjust as needed)
    fl.server.start_server(config={"num_rounds": 3})
