"""
Flower Server for YOLOv8 Federated Training
FIXED VERSION - Ensures consistent model architecture across server and clients
"""

import os
import sys
import argparse
import logging
from typing import List, Tuple, Dict, Optional
from datetime import datetime

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Configure professional logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup professional logging configuration"""
    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    # Suppress verbose third-party logs
    logging.getLogger('flwr').setLevel(logging.WARNING)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Setup logging (will be reconfigured in main with file option)
setup_logging()
logger = logging.getLogger(__name__)

import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays

# Import model utilities (with error handling)
try:
    from models.model import get_model, model_to_ndarrays, ndarrays_to_model, save_model
    MODEL_UTILS_AVAILABLE = True
    logger.debug("Model utilities imported successfully")
except ImportError as e:
    logger.error(f"Could not import model utilities: {e}")
    MODEL_UTILS_AVAILABLE = False


def weighted_average(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Compute weighted average of metrics from all clients.
    Handles mixed types, missing metrics, and errors gracefully.
    """
    if not metrics_list:
        return {}

    # Extract all unique metric keys
    all_keys = set()
    for num_examples, metrics in metrics_list:
        if isinstance(metrics, dict):
            all_keys.update(metrics.keys())
    
    aggregated: Metrics = {}
    
    for key in all_keys:
        # Skip error messages
        if key == 'error':
            continue
        
        values = []
        weights = []
        
        for num_examples, metrics in metrics_list:
            if isinstance(metrics, dict):
                value = metrics.get(key)
                
                # Only include numeric values
                if value is not None and isinstance(value, (int, float)):
                    try:
                        val_float = float(value)
                        # Skip NaN and Inf
                        if val_float == val_float and abs(val_float) != float('inf'):
                            values.append(val_float)
                            weights.append(num_examples)
                    except (ValueError, TypeError):
                        continue
        
        # Calculate weighted average
        if values and weights:
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                aggregated[key] = float(weighted_sum / total_weight)
    
    return aggregated


class DetailedFedAvg(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy with detailed logging and robust error handling"""
    
    def __init__(self, checkpoint_dir: str = "server/checkpoints", 
                 model_size: str = "n", num_classes: int = 7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.model_size = model_size
        self.num_classes = num_classes
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.round_num = 0
        self.expected_param_count = None  # Will be set from first client response

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate training results and save checkpoint"""
        self.round_num = server_round
        
        logger.info("="*80)
        logger.info(f"ROUND {server_round} - TRAINING AGGREGATION")
        logger.info("="*80)
        
        # Log failures
        if failures:
            logger.warning(f"{len(failures)} client(s) failed in round {server_round}")
            for failure in failures:
                logger.warning(f"Client failure: {failure}")
        
        # Log successful results
        if results:
            logger.info(f"Received training results from {len(results)} client(s)")
            
            total_samples = 0
            for idx, (client, fit_res) in enumerate(results, 1):
                num_samples = fit_res.num_examples
                metrics = fit_res.metrics
                param_count = len(parameters_to_ndarrays(fit_res.parameters))
                total_samples += num_samples
                
                # Verify parameter count consistency
                if self.expected_param_count is None:
                    self.expected_param_count = param_count
                    logger.info(f"Initialized expected parameter count: {param_count}")
                elif param_count != self.expected_param_count:
                    logger.error(
                        f"Parameter count mismatch for client {idx}: "
                        f"expected={self.expected_param_count}, received={param_count}"
                    )
                
                # Log client metrics
                if metrics:
                    numeric_metrics = {k: v for k, v in metrics.items() 
                                     if k != 'error' and isinstance(v, (int, float))}
                    logger.info(
                        f"Client {idx}: samples={num_samples}, "
                        f"params={param_count}, metrics={numeric_metrics}"
                    )
                else:
                    logger.info(f"Client {idx}: samples={num_samples}, params={param_count}")
            
            logger.info(f"Total samples across all clients: {total_samples}")
        else:
            logger.error("No successful training results received")
            return None
        
        # Perform FedAvg
        logger.info("Performing Federated Averaging (FedAvg)")
        try:
            agg_result = super().aggregate_fit(server_round, results, failures)
            
            if agg_result is None:
                logger.error("Aggregation failed - no results to aggregate")
                return None

            parameters, metrics = agg_result
            
            # Verify aggregated parameters
            agg_param_count = len(parameters_to_ndarrays(parameters))
            logger.info(f"Aggregation complete: {agg_param_count} parameter arrays")
            
            if self.expected_param_count and agg_param_count != self.expected_param_count:
                logger.error(
                    f"Parameter count mismatch after aggregation: "
                    f"expected={self.expected_param_count}, got={agg_param_count}"
                )
            
            # Log aggregated metrics
            if metrics:
                logger.info("Aggregated metrics:")
                for key, val in metrics.items():
                    if isinstance(val, (int, float)):
                        logger.info(f"  {key}: {val:.6f}")
            
            # Save checkpoint
            self._save_checkpoint(server_round, parameters)

            return parameters, metrics
            
        except Exception as e:
            logger.exception(f"Aggregation error: {e}")
            return None
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results"""
        logger.info("="*80)
        logger.info(f"ROUND {server_round} - EVALUATION AGGREGATION")
        logger.info("="*80)
        
        if failures:
            logger.warning(f"{len(failures)} client(s) failed evaluation")
        
        if not results:
            logger.error("No evaluation results received")
            return None
        
        logger.info(f"Received evaluation results from {len(results)} client(s)")
        
        # Log evaluation results
        for idx, (client, eval_res) in enumerate(results, 1):
            num_samples = eval_res.num_examples
            loss = eval_res.loss
            metrics = eval_res.metrics
            
            if metrics:
                numeric_metrics = {k: v for k, v in metrics.items() 
                                 if isinstance(v, (int, float))}
                logger.info(
                    f"Client {idx}: samples={num_samples}, loss={loss:.6f}, "
                    f"metrics={numeric_metrics}"
                )
            else:
                logger.info(f"Client {idx}: samples={num_samples}, loss={loss:.6f}")
        
        # Perform aggregation
        try:
            agg_result = super().aggregate_evaluate(server_round, results, failures)
            
            if agg_result:
                agg_loss, agg_metrics = agg_result
                logger.info(f"Federated average loss: {agg_loss:.6f}")
                if agg_metrics:
                    logger.info("Federated average metrics:")
                    for key, val in agg_metrics.items():
                        if isinstance(val, (int, float)):
                            logger.info(f"  {key}: {val:.6f}")
            
            return agg_result
            
        except Exception as e:
            logger.exception(f"Evaluation aggregation error: {e}")
            return None
    
    def _save_checkpoint(self, round_num, parameters):
        """Save model checkpoint (with error handling)"""
        if not MODEL_UTILS_AVAILABLE:
            logger.warning("Skipping checkpoint save (model utils not available)")
            return
        
        try:
            ndarrays = parameters_to_ndarrays(parameters)
            
            logger.info(f"Saving checkpoint for round {round_num}")
            
            # CRITICAL FIX: Use SAME model config as clients!
            logger.debug(f"Creating model: size={self.model_size}, classes={self.num_classes}")
            model = get_model(model_size=self.model_size, num_classes=self.num_classes, pretrained=True)
            
            # Count expected parameters from fresh model
            expected_arrays = len(model_to_ndarrays(model))
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.debug(
                f"Model parameters: trainable={trainable_params:,}, "
                f"expected_arrays={expected_arrays}, received_arrays={len(ndarrays)}"
            )
            
            # Verify counts match
            if len(ndarrays) != expected_arrays:
                logger.error(
                    f"Parameter count mismatch: expected={expected_arrays}, "
                    f"got={len(ndarrays)}. Skipping checkpoint save."
                )
                return
            
            # Load parameters
            ndarrays_to_model(model, ndarrays)
            
            # Save round checkpoint
            ckpt_path = os.path.join(self.checkpoint_dir, f"global_round_{round_num}.pt")
            save_model(model, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")
            
            # Save final model
            if hasattr(self, 'total_rounds') and round_num == self.total_rounds:
                final_path = os.path.join(project_root, "global_final_model.pt")
                save_model(model, final_path)
                logger.info(f"Final model saved: {final_path}")
                
        except Exception as e:
            logger.exception(f"Checkpoint save failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Flower Server for YOLOv8 Federated Learning")
    parser.add_argument("--addr", type=str, default="0.0.0.0:8080", help="Server address")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("--min-clients", type=int, default=3, help="Minimum clients")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients to fit")
    parser.add_argument("--model-size", type=str, default="n", help="Model size (n/s/m/l/x)")
    parser.add_argument("--num-classes", type=int, default=7, help="Number of classes")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--log-file", type=str, default=None, help="Log file path (optional)")
    args = parser.parse_args()

    # Setup logging with file option
    log_level = getattr(logging, args.log_level.upper())
    log_file = args.log_file or os.path.join("server", "logs", f"server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(log_level=log_level, log_file=log_file)
    
    logger.info("="*80)
    logger.info("FEDERATED LEARNING SERVER - YOLOv8 OBJECT DETECTION")
    logger.info("="*80)
    logger.info("Configuration:")
    logger.info(f"  Server Address: {args.addr}")
    logger.info(f"  Training Rounds: {args.rounds}")
    logger.info(f"  Minimum Clients: {args.min_clients}")
    logger.info(f"  Fraction Fit: {args.fraction_fit}")
    logger.info(f"  Model Size: YOLOv8{args.model_size}")
    logger.info(f"  Number of Classes: {args.num_classes}")
    logger.info(f"  Checkpoint Directory: server/checkpoints/")
    if args.log_file:
        logger.info(f"  Log File: {log_file}")
    logger.info("="*80)

    # Try to create initial parameters
    initial_parameters = None
    if MODEL_UTILS_AVAILABLE:
        logger.info("Initializing YOLOv8 model")
        try:
            # CRITICAL FIX: Use command-line args for consistent configuration
            model = get_model(model_size=args.model_size, num_classes=args.num_classes, pretrained=True)
            ndarrays = model_to_ndarrays(model)
            initial_parameters = ndarrays_to_parameters(ndarrays)
            
            total_weights = sum(arr.size for arr in ndarrays)
            logger.info(
                f"Initial parameters created: model=YOLOv8{args.model_size}, "
                f"classes={args.num_classes}, param_arrays={len(ndarrays)}, "
                f"total_weights={total_weights:,}"
            )
            
            # Log first 5 parameter shapes for debugging
            logger.debug("First 5 parameter array shapes:")
            for i in range(min(5, len(ndarrays))):
                logger.debug(f"  [{i}] {ndarrays[i].shape}")
            
        except Exception as e:
            logger.warning(f"Could not create initial parameters: {e}")
            logger.warning("Server will use client-initialized parameters")
            logger.debug("", exc_info=True)
    else:
        logger.warning("Model utilities not available - server will use client weights")

    # Create strategy
    logger.info("Configuring Federated Learning Strategy (FedAvg)")
    strategy = DetailedFedAvg(
        checkpoint_dir="server/checkpoints",
        model_size=args.model_size,
        num_classes=args.num_classes,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    strategy.total_rounds = args.rounds
    logger.info("Strategy configured: FedAvg with weighted metrics aggregation")

    logger.info("="*80)
    logger.info("Starting Flower Server")
    logger.info(f"Listening on: {args.addr}")
    logger.info(f"Waiting for {args.min_clients} client(s) to connect...")
    logger.info("="*80)

    # Start server
    try:
        fl.server.start_server(
            server_address=args.addr,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
        
        logger.info("="*80)
        logger.info("FEDERATED TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Completed {args.rounds} round(s) successfully")
        logger.info(f"Checkpoints saved in: server/checkpoints/")
        logger.info("="*80)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()