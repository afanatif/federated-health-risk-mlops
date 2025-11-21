# server/server_flower.py
"""
Version-adaptive Flower server starter.
Detects FedAvg constructor and start_server signature, calls appropriate API.
"""
import flwr as fl
import inspect
import sys
import traceback

def make_fedavg_strategy(num_clients_expected=3):
    """Create a FedAvg strategy while adapting to constructor signature changes."""
    FedAvg = fl.server.strategy.FedAvg
    sig = inspect.signature(FedAvg.__init__)
    params = sig.parameters

    kwargs = {}
    # choose candidates to set if supported
    candidates = {
        "fraction_fit": 1.0,
        "fraction_eval": 1.0,
        "min_fit_clients": num_clients_expected,
        "min_eval_clients": num_clients_expected,
        "min_available_clients": num_clients_expected,
    }
    for name, val in candidates.items():
        if name in params:
            kwargs[name] = val

    try:
        strategy = FedAvg(**kwargs) if kwargs else FedAvg()
        print(f"FedAvg created with args: {kwargs}")
        return strategy
    except Exception:
        print("Failed to create FedAvg with kwargs:", kwargs)
        traceback.print_exc()
        # Try fallback: bare FedAvg()
        try:
            strategy = FedAvg()
            print("FedAvg created with no args as fallback.")
            return strategy
        except Exception:
            print("Could not instantiate FedAvg at all.")
            raise

def start_server_agnostic(num_rounds: int = 3, address: str = "localhost:8080"):
    """
    Try different start_server signatures:
      - If ServerConfig exists, create it and pass config + strategy (newest)
      - Else if start_server accepts num_rounds keyword, use it
      - Else fallback to passing config as dict (older)
    """
    strategy = make_fedavg_strategy(num_clients_expected=3)

    start_sig = inspect.signature(fl.server.start_server)
    start_params = start_sig.parameters

    # Try ServerConfig (if available)
    ServerConfig = None
    # Common locations
    candidates = [
        ("flwr.server.server_config", "ServerConfig"),
        ("flwr.server", "ServerConfig"),
        ("flwr.server.server", "ServerConfig"),
    ]
    for modname, clsname in candidates:
        try:
            mod = __import__(modname, fromlist=[clsname])
            ServerConfig = getattr(mod, clsname)
            break
        except Exception:
            ServerConfig = None

    if "config" in start_params and ServerConfig is not None:
        try:
            cfg = ServerConfig(num_rounds=num_rounds)
            print("Starting Flower server using ServerConfig + strategy...")
            fl.server.start_server(server_address=address, config=cfg, strategy=strategy)
            return
        except Exception:
            traceback.print_exc()

    # If start_server supports num_rounds directly:
    if "num_rounds" in start_params:
        try:
            print("Starting Flower server using keyword num_rounds...")
            # pass strategy if accepted
            if "strategy" in start_params:
                fl.server.start_server(server_address=address, num_rounds=num_rounds, strategy=strategy)
            else:
                fl.server.start_server(server_address=address, num_rounds=num_rounds)
            return
        except TypeError:
            traceback.print_exc()
        except Exception:
            traceback.print_exc()

    # Fallback: pass config as dict (older versions)
    if "config" in start_params:
        try:
            print("Starting Flower server using dict config fallback...")
            fl.server.start_server(server_address=address, config={"num_rounds": num_rounds}, strategy=strategy)
            return
        except Exception:
            traceback.print_exc()

    print("All attempts to start Flower server failed.")
    sys.exit(1)


if __name__ == "__main__":
    start_server_agnostic(num_rounds=3, address="localhost:8080")
