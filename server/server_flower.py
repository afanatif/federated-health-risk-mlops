# server/server_flower.py
import flwr as fl

if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        config={"num_rounds": 3}  # just pass as a dict
    )
