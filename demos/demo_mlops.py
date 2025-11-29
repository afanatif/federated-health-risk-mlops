"""
MLOps Demo with Simulated Federated Learning
Demonstrates MLflow and Prometheus integration without real training
"""

import os
import time
import random
from datetime import datetime
import mlflow
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# Initialize Prometheus metrics
training_loss = Gauge('fl_training_loss', 'Training loss per round')
validation_loss = Gauge('fl_validation_loss', 'Validation loss per round')
accuracy = Gauge('fl_accuracy', 'Model accuracy per round')
num_clients = Gauge('fl_num_clients', 'Number of participating clients')
round_duration = Histogram('fl_round_duration_seconds', 'Duration of training round')
total_rounds = Counter('fl_total_rounds', 'Total number of rounds completed')

# Configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PROMETHEUS_PORT = 8081
NUM_ROUNDS = 10
NUM_CLIENTS = 3

def simulate_client_training(round_num):
    """Simulate training on a client with realistic metrics"""
    # Simulate improving metrics over rounds
    base_loss = 2.5
    improvement_factor = 0.85 ** round_num
    
    # Add some randomness
    train_loss = base_loss * improvement_factor + random.uniform(-0.1, 0.1)
    val_loss = train_loss * 1.1 + random.uniform(-0.05, 0.05)
    acc = min(0.95, 0.3 + (round_num * 0.065) + random.uniform(-0.02, 0.02))
    
    return {
        'train_loss': max(0.1, train_loss),
        'val_loss': max(0.15, val_loss),
        'accuracy': max(0.0, acc),
        'num_samples': random.randint(800, 1200)
    }

def aggregate_client_metrics(client_results):
    """Aggregate metrics from multiple clients (weighted average)"""
    total_samples = sum(r['num_samples'] for r in client_results)
    
    avg_train_loss = sum(r['train_loss'] * r['num_samples'] for r in client_results) / total_samples
    avg_val_loss = sum(r['val_loss'] * r['num_samples'] for r in client_results) / total_samples
    avg_accuracy = sum(r['accuracy'] * r['num_samples'] for r in client_results) / total_samples
    
    return {
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'accuracy': avg_accuracy,
        'total_samples': total_samples
    }

def run_demo():
    """Run the federated learning demo"""
    
    # Start Prometheus metrics server
    print(f"🚀 Starting Prometheus metrics server on port {PROMETHEUS_PORT}")
    start_http_server(PROMETHEUS_PORT)
    print(f"   Metrics available at: http://localhost:{PROMETHEUS_PORT}/metrics")
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("federated-learning-demo")
    
    print(f"\n📊 MLflow tracking URI: {MLFLOW_URI}")
    print(f"   MLflow UI: {MLFLOW_URI}")
    print(f"\n{'='*60}")
    print(f"Starting Federated Learning Demo")
    print(f"{'='*60}\n")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log hyperparameters
        mlflow.log_params({
            "num_rounds": NUM_ROUNDS,
            "num_clients": NUM_CLIENTS,
            "model": "YOLOv8n",
            "aggregation": "FedAvg",
            "demo_mode": True
        })
        
        # Simulate federated rounds
        for round_num in range(1, NUM_ROUNDS + 1):
            round_start = time.time()
            
            print(f"\n{'─'*60}")
            print(f"⚡ ROUND {round_num}/{NUM_ROUNDS}")
            print(f"{'─'*60}")
            
            # Simulate client training
            client_results = []
            for client_id in range(1, NUM_CLIENTS + 1):
                print(f"   Client {client_id}: Training...")
                time.sleep(0.5)  # Simulate training time
                
                metrics = simulate_client_training(round_num)
                client_results.append(metrics)
                
                print(f"   Client {client_id}: Loss={metrics['train_loss']:.4f}, "
                      f"Acc={metrics['accuracy']:.4f}, Samples={metrics['num_samples']}")
            
            # Aggregate metrics
            aggregated = aggregate_client_metrics(client_results)
            round_time = time.time() - round_start
            
            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": aggregated['train_loss'],
                "val_loss": aggregated['val_loss'],
                "accuracy": aggregated['accuracy'],
                "total_samples": aggregated['total_samples'],
                "round_duration": round_time
            }, step=round_num)
            
            # Update Prometheus metrics
            training_loss.set(aggregated['train_loss'])
            validation_loss.set(aggregated['val_loss'])
            accuracy.set(aggregated['accuracy'])
            num_clients.set(NUM_CLIENTS)
            round_duration.observe(round_time)
            total_rounds.inc()
            
            # Print aggregated results
            print(f"\n   📈 AGGREGATED RESULTS:")
            print(f"      Train Loss:  {aggregated['train_loss']:.4f}")
            print(f"      Val Loss:    {aggregated['val_loss']:.4f}")
            print(f"      Accuracy:    {aggregated['accuracy']:.4f} ({aggregated['accuracy']*100:.2f}%)")
            print(f"      Duration:    {round_time:.2f}s")
            
            time.sleep(1)  # Pause between rounds
        
        # Log final model metadata
        mlflow.log_param("final_accuracy", aggregated['accuracy'])
        mlflow.log_param("final_loss", aggregated['train_loss'])
        
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Final Accuracy: {aggregated['accuracy']:.4f} ({aggregated['accuracy']*100:.2f}%)")
        print(f"Final Loss: {aggregated['train_loss']:.4f}")
        print(f"\n📊 View results in MLflow: {MLFLOW_URI}")
        print(f"📈 View metrics in Prometheus: http://localhost:{PROMETHEUS_PORT}/metrics")
        print(f"\n{'='*60}\n")

if __name__ == "__main__":
    try:
        run_demo()
        
        # Keep server running to serve metrics
        print("⏳ Keeping metrics server running...")
        print("   Press Ctrl+C to stop\n")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
