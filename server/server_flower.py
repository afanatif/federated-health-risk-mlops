"""
Enhanced Flower Server with Detailed Metrics Display
Shows everything: round accuracy, federated averaging, weight changes, etc.
"""
import flwr as fl
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import Metrics, Parameters, Scalar, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy
import numpy as np
from datetime import datetime
import json


class DetailedMetricsLogger:
    """Logger to track and display detailed federated learning metrics."""
    
    def __init__(self):
        self.round_data = {}
        self.weight_stats = {}
        
    def log_round_start(self, round_num: int):
        """Log the start of a training round."""
        print("\n" + "="*80)
        print(f"{'ROUND ' + str(round_num) + ' STARTED':^80}")
        print("="*80)
        print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
        print("-"*80)
        
    def log_fit_results(self, round_num: int, results: List[Tuple[ClientProxy, FitRes]]):
        """Log detailed training results from all clients."""
        print(f"\n📊 ROUND {round_num} - TRAINING RESULTS FROM ALL CLIENTS:")
        print("-"*80)
        print(f"{'Client':<15} {'Samples':<10} {'Loss':<15} {'Metrics':<40}")
        print("-"*80)
        
        total_samples = 0
        losses = []
        all_metrics = {}
        
        for idx, (client, fit_res) in enumerate(results, 1):
            num_samples = fit_res.num_examples
            total_samples += num_samples
            
            # Extract loss from metrics if available
            loss = fit_res.metrics.get('loss', 'N/A')
            if isinstance(loss, (int, float)):
                losses.append(loss)
            
            # Format metrics
            metrics_str = str({k: f"{v:.4f}" if isinstance(v, float) else v 
                             for k, v in fit_res.metrics.items()})
            
            print(f"Client {idx:<8} {num_samples:<10} {loss:<15} {metrics_str:<40}")
            
            # Aggregate metrics
            for key, value in fit_res.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append((num_samples, value))
        
        print("-"*80)
        print(f"{'TOTAL':<15} {total_samples:<10}")
        
        # Calculate weighted averages
        if losses:
            avg_loss = sum(losses) / len(losses)
            print(f"\n📉 Average Training Loss: {avg_loss:.6f}")
        
        # Store for later
        self.round_data[round_num] = {
            'fit': {
                'total_samples': total_samples,
                'avg_loss': avg_loss if losses else None,
                'metrics': all_metrics
            }
        }
        
    def log_evaluate_results(self, round_num: int, results: List[Tuple[ClientProxy, EvaluateRes]]):
        """Log detailed evaluation results from all clients."""
        print(f"\n🎯 ROUND {round_num} - EVALUATION RESULTS FROM ALL CLIENTS:")
        print("-"*80)
        print(f"{'Client':<15} {'Samples':<10} {'Loss':<15} {'Accuracy':<15} {'Other Metrics':<30}")
        print("-"*80)
        
        total_samples = 0
        weighted_loss = 0
        weighted_accuracy = 0
        all_accuracies = []
        
        for idx, (client, eval_res) in enumerate(results, 1):
            num_samples = eval_res.num_examples
            loss = eval_res.loss
            accuracy = eval_res.metrics.get('accuracy', 0.0)
            
            total_samples += num_samples
            weighted_loss += loss * num_samples
            weighted_accuracy += accuracy * num_samples
            all_accuracies.append(accuracy)
            
            # Format other metrics
            other_metrics = {k: v for k, v in eval_res.metrics.items() if k != 'accuracy'}
            other_str = str({k: f"{v:.4f}" if isinstance(v, float) else v 
                           for k, v in other_metrics.items()})
            
            print(f"Client {idx:<8} {num_samples:<10} {loss:<15.6f} {accuracy*100:<14.2f}% {other_str:<30}")
        
        # Calculate federated averages
        fed_avg_loss = weighted_loss / total_samples if total_samples > 0 else 0
        fed_avg_accuracy = weighted_accuracy / total_samples if total_samples > 0 else 0
        
        print("-"*80)
        print(f"{'FEDERATED AVG':<15} {total_samples:<10} {fed_avg_loss:<15.6f} {fed_avg_accuracy*100:<14.2f}%")
        print("="*80)
        
        # Show improvement
        if round_num > 1 and (round_num - 1) in self.round_data:
            prev_acc = self.round_data[round_num - 1].get('eval', {}).get('fed_avg_accuracy', 0)
            improvement = (fed_avg_accuracy - prev_acc) * 100
            arrow = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            print(f"\n{arrow} Accuracy Change: {improvement:+.2f} percentage points")
        
        # Store for later
        if round_num not in self.round_data:
            self.round_data[round_num] = {}
        self.round_data[round_num]['eval'] = {
            'total_samples': total_samples,
            'fed_avg_loss': fed_avg_loss,
            'fed_avg_accuracy': fed_avg_accuracy,
            'individual_accuracies': all_accuracies
        }
    
    def log_aggregation(self, round_num: int, num_clients: int):
        """Log federated averaging process."""
        print(f"\n🔄 FEDERATED AVERAGING (Round {round_num}):")
        print("-"*80)
        print(f"📥 Received model updates from {num_clients} clients")
        print(f"⚙️  Computing weighted average of parameters...")
        print(f"   Formula: new_weight = Σ(client_weight × num_samples) / total_samples")
        print(f"📤 Broadcasting updated global model to all clients")
        
    def log_weight_statistics(self, round_num: int, parameters: Parameters):
        """Log statistics about model weights."""
        print(f"\n📊 MODEL WEIGHT STATISTICS (Round {round_num}):")
        print("-"*80)
        
        arrays = [np.array(p) for p in parameters.tensors]
        
        total_params = sum(arr.size for arr in arrays)
        print(f"Total Parameters: {total_params:,}")
        print(f"Number of Layers: {len(arrays)}")
        
        print(f"\n{'Layer':<10} {'Shape':<20} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
        print("-"*80)
        
        for idx, arr in enumerate(arrays[:5]):  # Show first 5 layers
            print(f"Layer {idx:<4} {str(arr.shape):<20} "
                  f"{arr.mean():<15.6f} {arr.std():<15.6f} "
                  f"{arr.min():<15.6f} {arr.max():<15.6f}")
        
        if len(arrays) > 5:
            print(f"... and {len(arrays) - 5} more layers")
    
    def print_final_summary(self):
        """Print final summary of all rounds."""
        print("\n\n" + "="*80)
        print(f"{'FINAL TRAINING SUMMARY':^80}")
        print("="*80)
        
        print(f"\n{'Round':<10} {'Train Loss':<15} {'Eval Loss':<15} {'Accuracy':<15} {'Improvement':<15}")
        print("-"*80)
        
        prev_acc = None
        for round_num in sorted(self.round_data.keys()):
            data = self.round_data[round_num]
            
            train_loss = data.get('fit', {}).get('avg_loss', 'N/A')
            eval_loss = data.get('eval', {}).get('fed_avg_loss', 'N/A')
            accuracy = data.get('eval', {}).get('fed_avg_accuracy', 0.0)
            
            improvement = ""
            if prev_acc is not None:
                diff = (accuracy - prev_acc) * 100
                arrow = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                improvement = f"{arrow} {diff:+.2f}%"
            
            train_loss_str = f"{train_loss:.6f}" if isinstance(train_loss, float) else str(train_loss)
            eval_loss_str = f"{eval_loss:.6f}" if isinstance(eval_loss, float) else str(eval_loss)
            accuracy_str = f"{accuracy*100:.2f}%"
            
            print(f"{round_num:<10} {train_loss_str:<15} {eval_loss_str:<15} {accuracy_str:<15} {improvement:<15}")
            
            prev_acc = accuracy
        
        print("="*80)
        
        # Best round
        if self.round_data:
            best_round = max(self.round_data.keys(), 
                           key=lambda r: self.round_data[r].get('eval', {}).get('fed_avg_accuracy', 0))
            best_acc = self.round_data[best_round]['eval']['fed_avg_accuracy']
            print(f"\n🏆 BEST ROUND: Round {best_round} with {best_acc*100:.2f}% accuracy")
        
        print("\n" + "="*80)


# Initialize logger
logger = DetailedMetricsLogger()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from multiple clients using weighted average.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples
    
    Returns:
        Aggregated metrics dictionary
    """
    if not metrics:
        return {}
    
    total_examples = sum(num for num, _ in metrics)
    
    if total_examples == 0:
        return {}
    
    # Initialize aggregated metrics
    aggregated = {}
    
    # Get all metric keys
    all_keys = set()
    for _, m in metrics:
        all_keys.update(m.keys())
    
    # Aggregate each metric with weighted average
    for key in all_keys:
        weighted_sum = sum(
            num * m.get(key, 0) 
            for num, m in metrics
        )
        aggregated[key] = weighted_sum / total_examples
    
    return aggregated


class DetailedFedAvg(fl.server.strategy.FedAvg):
    """Extended FedAvg strategy with detailed logging."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results with detailed logging."""
        
        self.current_round = server_round
        logger.log_round_start(server_round)
        
        # Log failures if any
        if failures:
            print(f"\n⚠️  WARNING: {len(failures)} clients failed during training")
        
        # Log individual client results
        logger.log_fit_results(server_round, results)
        
        # Perform standard aggregation
        logger.log_aggregation(server_round, len(results))
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Log weight statistics
        if aggregated_parameters:
            logger.log_weight_statistics(server_round, aggregated_parameters)
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results with detailed logging."""
        
        # Log failures if any
        if failures:
            print(f"\n⚠️  WARNING: {len(failures)} clients failed during evaluation")
        
        # Log individual client results
        logger.log_evaluate_results(server_round, results)
        
        # Perform standard aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        return aggregated_loss, aggregated_metrics


if __name__ == "__main__":
    print("\n" + "="*80)
    print(f"{'FEDERATED LEARNING SERVER - DETAILED METRICS MODE':^80}")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   • Strategy: FedAvg (Federated Averaging)")
    print(f"   • Number of Rounds: 3")
    print(f"   • Minimum Clients: 3")
    print(f"   • Server Address: localhost:8080")
    print(f"   • Metrics Tracking: ENABLED")
    print(f"   • Detailed Logging: ENABLED")
    print("\n" + "="*80)
    
    # Create detailed strategy
    strategy = DetailedFedAvg(
        fraction_fit=1.0,                          # Use all available clients for training
        fraction_evaluate=1.0,                     # Use all available clients for evaluation
        min_fit_clients=3,                         # Minimum clients for training
        min_evaluate_clients=3,                    # Minimum clients for evaluation
        min_available_clients=3,                   # Minimum clients that must connect
        fit_metrics_aggregation_fn=weighted_average,      # Aggregate training metrics
        evaluate_metrics_aggregation_fn=weighted_average, # Aggregate evaluation metrics
    )
    
    print(f"\n🚀 Starting Flower server...")
    print(f"⏳ Waiting for clients to connect...\n")
    
    # Start server
    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
    
    # Print final summary
    logger.print_final_summary()
    
    # Save metrics to file
    print("\n💾 Saving metrics to file: training_metrics.json")
    with open("training_metrics.json", "w") as f:
        json.dump(logger.round_data, f, indent=2, default=str)
    
    print("\n✅ Federated learning completed successfully!")
    print("="*80 + "\n")
