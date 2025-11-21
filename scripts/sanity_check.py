# scripts/sanity_check.py
import importlib
import socket
import sys
import os

print("=== Sanity check for federated-health-risk-mlops ===")
print(f"Working dir: {os.getcwd()}")

# 1) Check Python path / repo visibility
print("\n1) PYTHONPATH / package visibility")
try:
    import clients
    print(" - 'clients' package import: OK")
except Exception as e:
    print(" - 'clients' import FAILED:", e)

# 2) Check model import (use pretrained=False to avoid downloads)
print("\n2) Model import & instantiation")
try:
    model_mod = importlib.import_module("model")
    get_model = getattr(model_mod, "get_model")
    m = get_model(pretrained=False)
    print(" - model.get_model(pretrained=False): OK ->", type(m))
except Exception as e:
    print(" - model import failed:", repr(e))

# 3) Check data loaders for node1..3
print("\n3) Data loaders for node1/node2/node3 (attempt common signatures)")
for n in (1,2,3):
    ok = False
    node_str = f"clients.node{n}.data_loader"
    print(f" - node{n}: trying {node_str}.get_loaders ...")
    try:
        dlmod = importlib.import_module(node_str)
        if hasattr(dlmod, "get_loaders"):
            fn = dlmod.get_loaders
            # Try a few common calls; catch and print result
            try:
                loaders = fn(os.path.join("clients", f"node{n}", "data"))
                print(f"   * get_loaders(node_dir) -> loaders OK, types: {type(loaders)}")
                ok = True
            except Exception:
                try:
                    csv = os.path.join("clients", f"node{n}", "data", "sample.csv")
                    loaders = fn(csv)
                    print("   * get_loaders(csv_path) -> OK")
                    ok = True
                except Exception:
                    try:
                        imgs = os.path.join("clients", f"node{n}", "data", "images")
                        labs = os.path.join("clients", f"node{n}", "data", "labels")
                        loaders = fn(images_dir=imgs, labels_dir=labs, batch_size=8)
                        print("   * get_loaders(images_dir=..., labels_dir=..., batch_size=...) -> OK")
                        ok = True
                    except Exception as e:
                        print("   * get_loaders call attempts failed:", repr(e))
        else:
            print("   * get_loaders not found in module")
    except Exception as e:
        print(f"   * import failed: {e}")
    if not ok:
        print(f"   -> node{n} loader not OK. Fix get_loaders signature or placement.")
    else:
        print(f"   -> node{n} loader OK")

# 4) Check flwr version and available server.start_server signature
print("\n4) Flower (flwr) version & available functions")
try:
    import flwr as fl
    print(" - flwr.__version__:", fl.__version__)
    print(" - flwr.server attributes:", [a for a in dir(fl.server) if not a.startswith("_")][:50])
except Exception as e:
    print(" - flwr import failed:", e)

# 5) Check port 8080 availability (not that server is running)
print("\n5) Check port 8080 on localhost (server should bind to 8080 when running)")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.settimeout(0.5)
    res = s.connect_ex(("127.0.0.1", 8080))
    if res == 0:
        print(" - PORT 8080: IN USE (server might be running)")
    else:
        print(" - PORT 8080: free (server not running yet)")
finally:
    s.close()

print("\nSanity check complete. If model or loaders failed, fix those first.")
