"""
Flower Server for YOLOv8 Federated Training
FIXED VERSION - Ensures consistent model architecture across server and clients
"""

import os
import sys
import argparse
import logging
from typing import List, Tuple, Dict, Optional

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

print(f"Project Root: {project_root}")

import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays

# Import model utilities (with error handling)
try:
    from models.model import get_model, model_to_ndarrays, ndarrays_to_model, save_model
    MODEL_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Could not import model utilities: {e}")
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
                 model_size: str = "n", num_classes: int = 1, *args, **kwargs):
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
        
        print(f"\n{'='*80}")
        print(f"{'ROUND ' + str(server_round) + ' - TRAINING AGGREGATION':^80}")
        print(f"{'='*80}")
        
        # Show failures
        if failures:
            print(f"\n⚠️  {len(failures)} clients failed:")
            for failure in failures:
                print(f"   {failure}")
        
        # Show successful results
        if results:
            print(f"\n📊 Training Results from {len(results)} Clients:")
            print("-"*80)
            print(f"{'Client':<15} {'Samples':<15} {'Params':<15} {'Metrics':<35}")
            print("-"*80)
            
            for idx, (client, fit_res) in enumerate(results, 1):
                num_samples = fit_res.num_examples
                metrics = fit_res.metrics
                param_count = len(parameters_to_ndarrays(fit_res.parameters))
                
                # Verify parameter count consistency
                if self.expected_param_count is None:
                    self.expected_param_count = param_count
                    print(f"📌 Set expected parameter count: {param_count}")
                elif param_count != self.expected_param_count:
                    print(f"\n🚨 WARNING: Client {idx} parameter count mismatch!")
                    print(f"   Expected: {self.expected_param_count}, Got: {param_count}")
                    print(f"   This indicates model architecture inconsistency!")
                
                # Format metrics safely
                if metrics:
                    numeric_metrics = {k: v for k, v in metrics.items() 
                                     if k != 'error' and isinstance(v, (int, float))}
                    metrics_str = str(numeric_metrics)[:32] + "..." if len(str(numeric_metrics)) > 35 else str(numeric_metrics)
                else:
                    metrics_str = "No metrics"
                
                print(f"Client {idx:<8} {num_samples:<15} {param_count:<15} {metrics_str:<35}")
            
            print("-"*80)
            total_samples = sum(fit_res.num_examples for _, fit_res in results)
            print(f"{'TOTAL':<15} {total_samples:<15}")
        else:
            print(f"\n❌ No successful training results!")
            return None
        
        # Perform FedAvg
        print(f"\n🔄 Performing Federated Averaging...")
        try:
            agg_result = super().aggregate_fit(server_round, results, failures)
            
            if agg_result is None:
                print(f"❌ Aggregation failed - no results to aggregate")
                return None

            parameters, metrics = agg_result
            
            # Verify aggregated parameters
            agg_param_count = len(parameters_to_ndarrays(parameters))
            print(f"✅ Aggregation complete")
            print(f"   Aggregated parameter arrays: {agg_param_count}")
            
            if self.expected_param_count and agg_param_count != self.expected_param_count:
                print(f"🚨 CRITICAL: Aggregated param count ({agg_param_count}) != expected ({self.expected_param_count})")
                print(f"   This indicates parameter loss during aggregation!")
            
            # Show aggregated metrics
            if metrics:
                print(f"\n📊 Aggregated Metrics:")
                for key, val in metrics.items():
                    if isinstance(val, (int, float)):
                        print(f"   {key}: {val:.6f}")
            
            # Save checkpoint
            self._save_checkpoint(server_round, parameters)

            return parameters, metrics
            
        except Exception as e:
            print(f"❌ Aggregation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results"""
        print(f"\n{'='*80}")
        print(f"{'ROUND ' + str(server_round) + ' - EVALUATION AGGREGATION':^80}")
        print(f"{'='*80}")
        
        if failures:
            print(f"\n⚠️  {len(failures)} clients failed evaluation")
        
        if not results:
            print(f"❌ No evaluation results")
            return None
        
        # Show evaluation results
        print(f"\n🎯 Evaluation Results from {len(results)} Clients:")
        print("-"*80)
        print(f"{'Client':<15} {'Samples':<15} {'Loss':<15} {'Metrics':<30}")
        print("-"*80)
        
        for idx, (client, eval_res) in enumerate(results, 1):
            num_samples = eval_res.num_examples
            loss = eval_res.loss
            metrics = eval_res.metrics
            
            if metrics:
                numeric_metrics = {k: v for k, v in metrics.items() 
                                 if isinstance(v, (int, float))}
                metrics_str = str(numeric_metrics)[:27] + "..." if len(str(numeric_metrics)) > 30 else str(numeric_metrics)
            else:
                metrics_str = "No metrics"
            
            print(f"Client {idx:<8} {num_samples:<15} {loss:<15.6f} {metrics_str:<30}")
        
        print("-"*80)
        
        # Perform aggregation
        try:
            agg_result = super().aggregate_evaluate(server_round, results, failures)
            
            if agg_result:
                agg_loss, agg_metrics = agg_result
                print(f"\n📊 Federated Average:")
                print(f"   Loss: {agg_loss:.6f}")
                if agg_metrics:
                    for key, val in agg_metrics.items():
                        if isinstance(val, (int, float)):
                            print(f"   {key}: {val:.6f}")
            
            print("="*80)
            return agg_result
            
        except Exception as e:
            print(f"❌ Evaluation aggregation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_checkpoint(self, round_num, parameters):
        """Save model checkpoint (with error handling)"""
        if not MODEL_UTILS_AVAILABLE:
            print(f"⚠️  Skipping checkpoint save (model utils not available)")
            return
        
        try:
            ndarrays = parameters_to_ndarrays(parameters)
            
            print(f"\n💾 Saving checkpoint for round {round_num}...")
            
            # CRITICAL FIX: Use SAME model config as clients!
            print(f"   Creating model with: model_size='{self.model_size}', num_classes={self.num_classes}")
            model = get_model(model_size=self.model_size, num_classes=self.num_classes, pretrained=False)
            
            # Count expected parameters from fresh model
            expected_arrays = len(model_to_ndarrays(model))
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"   Model created:")
            print(f"     • Trainable parameters: {trainable_params:,}")
            print(f"     • Expected parameter arrays: {expected_arrays}")
            print(f"     • Received parameter arrays: {len(ndarrays)}")
            
            # Verify counts match
            if len(ndarrays) != expected_arrays:
                print(f"\n🚨 CRITICAL ERROR: Parameter count mismatch!")
                print(f"   Expected: {expected_arrays}, Got: {len(ndarrays)}")
                print(f"   Cannot safely load parameters - SKIPPING CHECKPOINT SAVE")
                print(f"   This indicates server and client models are different architectures!")
                return
            
            # Show shape comparison for first 5 arrays
            print(f"\n   First 5 parameter shapes:")
            model_arrays = model_to_ndarrays(model)
            for i in range(min(5, len(ndarrays))):
                match = "✓" if ndarrays[i].shape == model_arrays[i].shape else "✗"
                print(f"     [{i}] Server: {model_arrays[i].shape}, Received: {ndarrays[i].shape} {match}")
            
            # Load parameters
            ndarrays_to_model(model, ndarrays)
            
            # Save round checkpoint
            ckpt_path = os.path.join(self.checkpoint_dir, f"global_round_{round_num}.pt")
            save_model(model, ckpt_path)
            print(f"   ✅ Saved: {ckpt_path}")
            
            # Save final model
            if hasattr(self, 'total_rounds') and round_num == self.total_rounds:
                final_path = os.path.join(project_root, "global_final_model.pt")
                save_model(model, final_path)
                print(f"   🎉 Final model saved: {final_path}")
                
        except Exception as e:
            print(f"⚠️  Checkpoint save failed: {e}")
            print(f"   This may indicate parameter shape mismatches")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Flower Server for YOLOv8 Federated Learning")
    parser.add_argument("--addr", type=str, default="0.0.0.0:8080", help="Server address")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("--min-clients", type=int, default=3, help="Minimum clients")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients to fit")
    parser.add_argument("--model-size", type=str, default="n", help="Model size (n/s/m/l/x)")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of classes")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("FEDERATED LEARNING SERVER - YOLOv8 POTHOLE DETECTION".center(80))
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   • Server Address: {args.addr}")
    print(f"   • Training Rounds: {args.rounds}")
    print(f"   • Minimum Clients: {args.min_clients}")
    print(f"   • Fraction Fit: {args.fraction_fit}")
    print(f"   • Model Size: YOLOv8{args.model_size}")
    print(f"   • Number of Classes: {args.num_classes}")
    print(f"   • Checkpoints: server/checkpoints/")
    print("\n" + "="*80)

    # Try to create initial parameters
    initial_parameters = None
    if MODEL_UTILS_AVAILABLE:
        print("\n🤖 Initializing YOLOv8 Model...")
        try:
            # CRITICAL FIX: Use command-line args for consistent configuration
            model = get_model(model_size=args.model_size, num_classes=args.num_classes, pretrained=False)
            ndarrays = model_to_ndarrays(model)
            initial_parameters = ndarrays_to_parameters(ndarrays)
            
            print(f"✅ Initial parameters created:")
            print(f"   • Model: YOLOv8{args.model_size}")
            print(f"   • Classes: {args.num_classes}")
            print(f"   • Parameter arrays: {len(ndarrays)}")
            print(f"   • Total weights: {sum(arr.size for arr in ndarrays):,}")
            
            # Show first 5 parameter shapes for debugging
            print(f"\n   First 5 parameter array shapes:")
            for i in range(min(5, len(ndarrays))):
                print(f"     [{i}] {ndarrays[i].shape}")
            
        except Exception as e:
            print(f"⚠️  Could not create initial parameters: {e}")
            print(f"   Server will use client-initialized parameters")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️  Model utilities not available - server will use client weights")

    # Create strategy
    print("\n🔧 Configuring Federated Learning Strategy...")
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
    print(f"✅ Strategy configured: FedAvg with weighted metrics")

    print(f"\n🚀 Starting Flower Server...")
    print(f"📡 Listening on: {args.addr}")
    print(f"⏳ Waiting for {args.min_clients} clients to connect...")
    print("="*80 + "\n")

    # Start server
    fl.server.start_server(
        server_address=args.addr,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    
    print("\n" + "="*80)
    print("🏁 FEDERATED TRAINING COMPLETE!")
    print("="*80)
    print(f"\n✅ All {args.rounds} rounds completed successfully")
    print(f"📁 Checkpoints saved in: server/checkpoints/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()