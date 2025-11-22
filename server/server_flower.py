"""
Flower server for YOLOv8 federated training using Weighted FedAvg.

- Expects your repo to provide:
    models.model.get_model
    models.model.model_to_ndarrays
    models.model.ndarrays_to_model
    models.model.save_model

- Saves global model after each round to server/checkpoints/global_round_{round}.pt
"""

import os
import argparse
import logging
from typing import List, Tuple, Dict, Optional

import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays

# ----------------------------
# Weighted metrics aggregator
# ----------------------------
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: Metrics = {}
    # union of keys across clients
    metric_keys = set().union(*(m.keys() for _, m in metrics))
    for key in metric_keys:
        weighted_sum = sum(num_examples * m.get(key, 0.0) for num_examples, m in metrics)
        aggregated[key] = weighted_sum / total_examples
    return aggregated

# ----------------------------
# Custom FedAvg that saves global model each round
# ----------------------------
class SaveCheckpointFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, checkpoint_dir: str = "server/checkpoints", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Note: fraction_fit, min_fit_clients, etc. are handled by *args, **kwargs

    def aggregate_fit(self, rnd, results, failures):
        """
        Called after clients returned fit results. We call parent aggregator to perform FedAvg,
        then save the aggregated parameters to disk as a model checkpoint.
        """
        agg_result = super().aggregate_fit(rnd, results, failures)
        # agg_result is Optional[Tuple[Parameters, Metrics]]
        if agg_result is None:
            logging.warning("aggregate_fit returned None (no aggregation performed).")
            return None

        parameters, metrics = agg_result

        try:
            # Convert Parameters -> ndarrays
            ndarrays = parameters_to_ndarrays(parameters)
            # Reconstruct a model and load ndarrays, then save checkpoint
            from models.model import get_model, ndarrays_to_model, save_model

            model = get_model(model_size="n", num_classes=1, pretrained=False)
            ndarrays_to_model(model, ndarrays)
            
            # Save the round-specific checkpoint
            ckpt_path = os.path.join(self.checkpoint_dir, f"global_round_{rnd}.pt")
            save_model(model, ckpt_path)
            logging.info("Saved global checkpoint: %s", ckpt_path)

            # CRITICAL: Also save the final round model to a fixed name for artifact upload
            if rnd == self.config.num_rounds:
                final_path = os.path.join(os.getcwd(), "global_final_model.pt")
                save_model(model, final_path)
                logging.info("Saved final model to: %s for artifact upload.", final_path)

        except Exception as e:
            logging.exception("Failed to save checkpoint after aggregation: %s", e)

        return parameters, metrics

# ----------------------------
# Main server start
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", type=str, default="0.0.0.0:8080", help="Server bind address")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--min-clients", type=int, default=3, help="Minimum available clients")
    
    # ⭐ ADDED ARGUMENT: The fraction of clients required to participate in a round.
    # Set to a float value (e.g., 0.34 for 1/3 clients in CI/CD).
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of available clients required for fitting.")
    
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("=" * 60)
    logging.info("Starting Flower Federated Learning Server (Weighted FedAvg)")
    logging.info("Address: %s | Rounds: %d | Min clients: %d | Fraction Fit: %.2f", 
                 args.addr, args.rounds, args.min_clients, args.fraction_fit)
    logging.info("=" * 60)

    # Try to create initial parameters from your model (optional)
    initial_parameters = None
    try:
        from models.model import get_model, model_to_ndarrays
        logging.info("Building initial model parameters from models.model.get_model")
        model = get_model(model_size="n", num_classes=1, pretrained=False)
        ndarrays = model_to_ndarrays(model)
        initial_parameters = ndarrays_to_parameters(ndarrays)
        logging.info("Initial parameters created (length=%d ndarrays).", len(ndarrays))
    except Exception as e:
        logging.warning("Could not build initial parameters from models.model: %s", e)
        logging.info("Server will rely on client-initialized parameters if needed.")

    # Prepare strategy
    strategy = SaveCheckpointFedAvg(
        checkpoint_dir="server/checkpoints",
        
        # ⭐ CRITICAL CHANGE: Use the parsed argument for fraction_fit
        fraction_fit=args.fraction_fit,
        
        fraction_evaluate=1.0, # Keep this high or match fraction_fit
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start server (blocking)
    fl.server.start_server(
        server_address=args.addr,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
