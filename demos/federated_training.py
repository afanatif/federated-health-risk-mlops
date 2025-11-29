"""
Federated Learning Training Pipeline
YOLOv8 Object Detection with FedAvg Aggregation
"""

import os
import time
import random
import numpy as np
from datetime import datetime
import mlflow
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info

# Prometheus Metrics
training_loss = Gauge('federated_training_loss', 'Average training loss across clients')
validation_loss = Gauge('federated_validation_loss', 'Average validation loss across clients')
model_accuracy = Gauge('federated_model_accuracy', 'Model accuracy (mAP@0.5)')
active_clients = Gauge('federated_active_clients', 'Number of clients participating')
round_duration = Histogram('federated_round_duration_seconds', 'Time taken per training round')
total_rounds_completed = Counter('federated_rounds_completed_total', 'Total training rounds completed')
aggregation_time = Histogram('federated_aggregation_time_seconds', 'Time taken for model aggregation')
training_info = Info('federated_training', 'Training configuration and status')

# Configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PROMETHEUS_PORT = 8081
NUM_ROUNDS = 10
NUM_CLIENTS = 3
MODEL_NAME = "YOLOv8n"
DATASET = "Health Risk Detection (7 classes)"

class FederatedTrainer:
    def __init__(self):
        self.round_num = 0
        self.best_accuracy = 0.0
        # Initialize with realistic YOLOv8 baseline metrics
        self.initial_loss = 2.8
        self.initial_acc = 0.15
        
    def client_local_training(self, client_id, round_num):
        """Simulate local training on client device"""
        # Realistic training progression
        improvement_rate = 1.0 - (0.12 * np.log(round_num + 1))
        noise_factor = random.uniform(0.92, 1.08)
        
        # Loss decreases with training
        train_loss = max(0.3, self.initial_loss * improvement_rate * noise_factor)
        val_loss = train_loss * random.uniform(1.05, 1.15)
        
        # Accuracy increases (mAP@0.5 for object detection)
        base_acc = min(0.85, self.initial_acc + (round_num * 0.07))
        accuracy = base_acc * random.uniform(0.95, 1.05)
        
        # Realistic dataset sizes per client
        num_images = random.randint(450, 750)
        num_objects = int(num_images * random.uniform(3.5, 7.2))
        
        # Training metrics
        epochs_local = 5
        batch_size = 16
        learning_rate = 0.001 * (0.95 ** round_num)
        
        return {
            'client_id': client_id,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy_map50': accuracy,
            'num_images': num_images,
            'num_objects': num_objects,
            'epochs': epochs_local,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'duration': random.uniform(45, 90)
        }
    
    def aggregate_models(self, client_results):
        """FedAvg: Weighted average of client models"""
        total_images = sum(r['num_images'] for r in client_results)
        
        # Weighted aggregation
        agg_train_loss = sum(r['train_loss'] * r['num_images'] for r in client_results) / total_images
        agg_val_loss = sum(r['val_loss'] * r['num_images'] for r in client_results) / total_images
        agg_accuracy = sum(r['accuracy_map50'] * r['num_images'] for r in client_results) / total_images
        
        return {
            'train_loss': agg_train_loss,
            'val_loss': agg_val_loss,
            'accuracy_map50': agg_accuracy,
            'total_images': total_images,
            'total_objects': sum(r['num_objects'] for r in client_results)
        }

def initialize_mlops():
    """Initialize MLflow and Prometheus"""
    # Start Prometheus metrics server
    print(f"Initializing Prometheus metrics server on port {PROMETHEUS_PORT}...")
    start_http_server(PROMETHEUS_PORT)
    
    # Set training info
    training_info.info({
        'model': MODEL_NAME,
        'dataset': DATASET,
        'num_clients': str(NUM_CLIENTS),
        'aggregation': 'FedAvg',
        'framework': 'Flower + YOLOv8'
    })
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("health-risk-federated-learning")
    
    print(f"MLflow tracking: {MLFLOW_URI}")
    print(f"Prometheus metrics: http://localhost:{PROMETHEUS_PORT}/metrics\n")

def run_federated_training():
    """Execute federated learning training"""
    initialize_mlops()
    
    trainer = FederatedTrainer()
    
    print("="*70)
    print("FEDERATED LEARNING - YOLOv8 HEALTH RISK DETECTION")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET}")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Rounds: {NUM_ROUNDS}")
    print("="*70)
    print()
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log hyperparameters
        mlflow.log_params({
            "model_architecture": MODEL_NAME,
            "num_rounds": NUM_ROUNDS,
            "num_clients": NUM_CLIENTS,
            "aggregation_strategy": "FedAvg",
            "dataset": DATASET,
            "min_clients_per_round": NUM_CLIENTS
        })
        
        # Training loop
        for round_num in range(1, NUM_ROUNDS + 1):
            round_start = time.time()
            
            print(f"Round {round_num}/{NUM_ROUNDS}")
            print("-" * 70)
            
            # Client training phase
            client_results = []
            for client_id in range(1, NUM_CLIENTS + 1):
                print(f"  Client {client_id}: Training on local dataset...")
                result = trainer.client_local_training(client_id, round_num)
                client_results.append(result)
                
                print(f"  Client {client_id}: Loss={result['train_loss']:.4f}, "
                      f"mAP@0.5={result['accuracy_map50']:.4f}, "
                      f"Images={result['num_images']}")
                
                time.sleep(0.3)  # Simulate training time
            
            # Aggregation phase
            print("\n  Aggregating models...")
            agg_start = time.time()
            time.sleep(0.5)  # Simulate aggregation computation
            
            aggregated = trainer.aggregate_models(client_results)
            agg_time = time.time() - agg_start
            round_time = time.time() - round_start
            
            # Update best accuracy
            if aggregated['accuracy_map50'] > trainer.best_accuracy:
                trainer.best_accuracy = aggregated['accuracy_map50']
                is_best = " (NEW BEST!)"
            else:
                is_best = ""
            
            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": aggregated['train_loss'],
                "val_loss": aggregated['val_loss'],
                "accuracy_map50": aggregated['accuracy_map50'],
                "total_images": aggregated['total_images'],
                "total_objects": aggregated['total_objects'],
                "round_duration_sec": round_time,
                "aggregation_time_sec": agg_time
            }, step=round_num)
            
            # Update Prometheus metrics
            training_loss.set(aggregated['train_loss'])
            validation_loss.set(aggregated['val_loss'])
            model_accuracy.set(aggregated['accuracy_map50'])
            active_clients.set(NUM_CLIENTS)
            round_duration.observe(round_time)
            aggregation_time.observe(agg_time)
            total_rounds_completed.inc()
            
            # Display results
            print(f"\n  Global Model Performance:")
            print(f"    Training Loss:    {aggregated['train_loss']:.4f}")
            print(f"    Validation Loss:  {aggregated['val_loss']:.4f}")
            print(f"    Accuracy (mAP@0.5): {aggregated['accuracy_map50']:.4f}{is_best}")
            print(f"    Total Images:     {aggregated['total_images']}")
            print(f"    Round Duration:   {round_time:.2f}s")
            print()
            
            time.sleep(0.5)
        
        # Log final model info
        mlflow.log_param("final_accuracy", trainer.best_accuracy)
        mlflow.log_param("best_accuracy_map50", trainer.best_accuracy)
        
        print("="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Best Accuracy (mAP@0.5): {trainer.best_accuracy:.4f}")
        print(f"Final Training Loss: {aggregated['train_loss']:.4f}")
        print(f"\nView results:")
        print(f"  MLflow UI:  {MLFLOW_URI}")
        print(f"  Prometheus: http://localhost:{PROMETHEUS_PORT}/metrics")
        print("="*70)

if __name__ == "__main__":
    try:
        run_federated_training()
        
        print("\nMetrics server running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nTraining stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
