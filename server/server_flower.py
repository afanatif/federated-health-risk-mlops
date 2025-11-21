# server/server_flower.py
"""
Version-robust Flower server starter.
Tries a few common API signatures so it works across Flower releases.
"""
import flwr as fl
import sys
import traceback

def start_server_agnostic(num_rounds: int = 3, address: str = "localhost:8080"):
    # Try pattern 1: modern Flower with ServerConfig + strategy (>=1.6)
    try:
        # Strategy (FedAvg) - standard
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_eval=1.0,
            min_fit_clients=3,
            min_eval_clients=3,
            min_available_clients=3,
        )
        # Try different ServerConfig import locations
        try:
            from flwr.server.server_config import ServerConfig
            cfg = ServerConfig(num_rounds=num_rounds)
        except Exception:
            try:
                from flwr.server import ServerConfig
                cfg = ServerConfig(num_rounds=num_rounds)
            except Exception:
                cfg = None

        if cfg is not None:
            print("Starting server using ServerConfig + strategy...")
            fl.server.start_server(server_address=address, config=cfg, strategy=strategy)
            return
    except Exception:
        # Continue to next attempt
        traceback.print_exc()
        pass

    # Try pattern 2: mid API (num_rounds as keyword)
    try:
        print("Trying start_server(server_address=..., num_rounds=...) ...")
        fl.server.start_server(server_address=address, num_rounds=num_rounds)
        return
    except TypeError:
        # Not supported: try next
        pass
    except Exception:
        traceback.print_exc()

    # Try pattern 3: old API (config dict)
    try:
        print("Falling back to start_server(server_address=..., config={...}) ...")
        fl.server.start_server(server_address=address, config={"num_rounds": num_rounds})
        return
    except Exception as e:
        print("All start_server attempts failed.")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    start_server_agnostic(num_rounds=3, address="localhost:8080")
