"""Simple MLflow Demo - Pre-populate with realistic federated learning results"""
import mlflow
import random
import time

# Use MLflow from Docker
MLFLOW_URI = "http://fl-mlflow:5000"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("health-risk-federated-learning")

print("Starting demo training runs...")

# Create 3 realistic training runs
for run_num in range(1, 4):
    with mlflow.start_run(run_name=f"federated_run_{run_num}"):
        # Log hyperparameters
        mlflow.log_params({
            "model": "YOLOv8n",
            "num_clients": 3,
            "num_rounds": 10,
            "aggregation": "FedAvg",
            "dataset": "Health Risk Detection"
        })
        
        # Simulate 10 rounds of training with improving metrics
        for round in range(1, 11):
            # Realistic metric progression
            train_loss = 2.5 * (0.85 ** round) + random.uniform(-0.1, 0.1)
            val_loss = train_loss * 1.15
            accuracy = min(0.85, 0.2 + (round * 0.065)) + random.uniform(-0.02, 0.02)
            
            mlflow.log_metrics({
                "train_loss": max(0.3, train_loss),
                "val_loss": max(0.35, val_loss),
                "accuracy_map50": max(0.15, accuracy),
                "num_clients": 3,
                "round_duration": random.uniform(45, 90)
            }, step=round)
            
            time.sleep(0.1)
        
        # Log final metrics
        mlflow.log_param("best_accuracy", max(0.15, accuracy))
        
        print(f"✓ Run {run_num} completed")

print("\n✅ Demo complete! View results at http://localhost:5000")
