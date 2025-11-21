"""
Experiment Tracking with MLflow
Tracks: hyperparameters, metrics, models, artifacts
"""
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
import json
from datetime import datetime
from typing import Dict, Any, Optional
import os


class ExperimentTracker:
    """MLflow experiment tracker for federated learning."""
    
    def __init__(
        self,
        experiment_name: str = "federated-health-risk-mlops",
        tracking_uri: str = "file:./mlruns"
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                tags={
                    "project": "federated-learning",
                    "model_type": "resnet50",
                    "task": "pothole-detection"
                }
            )
        else:
            self.experiment_id = self.experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        self.run = None
    
    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for this training run
            tags: Additional tags for the run
        """
        default_tags = {
            "timestamp": datetime.now().isoformat(),
            "framework": "pytorch",
            "federated": "true"
        }
        
        if tags:
            default_tags.update(tags)
        
        self.run = mlflow.start_run(run_name=run_name, tags=default_tags)
        print(f"📊 Started MLflow run: {run_name}")
        print(f"   Run ID: {self.run.info.run_id}")
        return self.run
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of parameter names and values
        """
        if self.run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        print(f"✅ Logged {len(params)} parameters")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step/epoch number
        """
        if self.run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_federated_round(
        self,
        round_num: int,
        train_loss: float,
        eval_loss: float,
        accuracy: float,
        num_clients: int
    ):
        """
        Log federated learning round metrics.
        
        Args:
            round_num: Round number
            train_loss: Training loss
            eval_loss: Evaluation loss
            accuracy: Federated accuracy
            num_clients: Number of participating clients
        """
        metrics = {
            f"round_{round_num}_train_loss": train_loss,
            f"round_{round_num}_eval_loss": eval_loss,
            f"round_{round_num}_accuracy": accuracy,
            f"round_{round_num}_num_clients": num_clients,
            "train_loss": train_loss,  # Latest
            "eval_loss": eval_loss,     # Latest
            "accuracy": accuracy,        # Latest
        }
        
        self.log_metrics(metrics, step=round_num)
        
        print(f"📊 Round {round_num} metrics logged:")
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Eval Loss: {eval_loss:.6f}")
        print(f"   Accuracy: {accuracy*100:.2f}%")
    
    def log_client_metrics(
        self,
        round_num: int,
        client_id: int,
        metrics: Dict[str, float]
    ):
        """
        Log individual client metrics.
        
        Args:
            round_num: Round number
            client_id: Client/node identifier
            metrics: Client-specific metrics
        """
        for key, value in metrics.items():
            metric_name = f"client_{client_id}_{key}"
            mlflow.log_metric(metric_name, value, step=round_num)
    
    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """
        Log PyTorch model.
        
        Args:
            model: PyTorch model to log
            artifact_path: Path within the run's artifact directory
            registered_model_name: Name for model registry
        """
        if self.run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
        
        print(f"✅ Model logged to MLflow")
        if registered_model_name:
            print(f"   Registered as: {registered_model_name}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact file.
        
        Args:
            local_path: Local path to the artifact
            artifact_path: Path within the run's artifact directory
        """
        if self.run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        print(f"✅ Artifact logged: {local_path}")
    
    def log_dict(self, dictionary: Dict[str, Any], filename: str):
        """
        Log a dictionary as JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            filename: Name for the JSON file
        """
        if self.run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        # Save to temp file
        temp_path = f"/tmp/{filename}"
        with open(temp_path, 'w') as f:
            json.dump(dictionary, f, indent=2, default=str)
        
        # Log artifact
        mlflow.log_artifact(temp_path)
        print(f"✅ Dictionary logged as: {filename}")
    
    def log_figure(self, figure, filename: str):
        """
        Log a matplotlib figure.
        
        Args:
            figure: Matplotlib figure
            filename: Name for the figure file
        """
        if self.run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        temp_path = f"/tmp/{filename}"
        figure.savefig(temp_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(temp_path)
        print(f"✅ Figure logged: {filename}")
    
    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for the current run.
        
        Args:
            tags: Dictionary of tag names and values
        """
        if self.run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        for key, value in tags.items():
            mlflow.set_tag(key, value)
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if self.run is not None:
            mlflow.end_run(status=status)
            print(f"🏁 Run ended with status: {status}")
            self.run = None
    
    def compare_runs(self, run_ids: list) -> Dict[str, Any]:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
        
        Returns:
            Dictionary with comparison data
        """
        comparison = {}
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            comparison[run_id] = {
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags
            }
        
        return comparison
    
    def get_best_run(self, metric: str = "accuracy", mode: str = "max"):
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric name to optimize
            mode: 'max' or 'min'
        
        Returns:
            Best run information
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'DESC' if mode == 'max' else 'ASC'}"]
        )
        
        if len(runs) == 0:
            return None
        
        best_run = runs.iloc[0]
        print(f"🏆 Best run based on {metric}:")
        print(f"   Run ID: {best_run.run_id}")
        print(f"   {metric}: {best_run[f'metrics.{metric}']}")
        
        return best_run


class FederatedLearningTracker(ExperimentTracker):
    """Extended tracker specifically for federated learning."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_data = {}
    
    def log_federated_configuration(
        self,
        num_rounds: int,
        num_clients: int,
        strategy: str,
        model_name: str
    ):
        """Log federated learning configuration."""
        config = {
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "strategy": strategy,
            "model_name": model_name,
        }
        
        self.log_params(config)
        self.set_tags({
            "fl_strategy": strategy,
            "num_clients": str(num_clients)
        })
    
    def log_complete_round(
        self,
        round_num: int,
        aggregated_metrics: Dict[str, float],
        client_metrics: list
    ):
        """
        Log complete round information including all clients.
        
        Args:
            round_num: Round number
            aggregated_metrics: Federated metrics
            client_metrics: List of per-client metrics
        """
        # Log aggregated metrics
        for key, value in aggregated_metrics.items():
            mlflow.log_metric(f"federated_{key}", value, step=round_num)
        
        # Log per-client metrics
        for idx, client_metric in enumerate(client_metrics, 1):
            for key, value in client_metric.items():
                mlflow.log_metric(f"client_{idx}_{key}", value, step=round_num)
        
        # Store for later analysis
        self.round_data[round_num] = {
            "aggregated": aggregated_metrics,
            "clients": client_metrics
        }
        
        print(f"✅ Round {round_num} complete data logged")
    
    def create_training_summary(self):
        """Create and log a training summary."""
        if not self.round_data:
            print("⚠️  No round data to summarize")
            return
        
        summary = {
            "total_rounds": len(self.round_data),
            "best_accuracy": max(
                r["aggregated"].get("accuracy", 0)
                for r in self.round_data.values()
            ),
            "final_loss": list(self.round_data.values())[-1]["aggregated"].get("loss", 0),
            "rounds": self.round_data
        }
        
        self.log_dict(summary, "training_summary.json")
        print("✅ Training summary created and logged")


def demo_tracking():
    """Demonstrate MLflow tracking capabilities."""
    print("="*80)
    print("MLflow Experiment Tracking Demo")
    print("="*80)
    
    # Initialize tracker
    tracker = FederatedLearningTracker()
    
    # Start run
    tracker.start_run(
        run_name=f"demo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags={"demo": "true"}
    )
    
    # Log configuration
    tracker.log_federated_configuration(
        num_rounds=3,
        num_clients=3,
        strategy="FedAvg",
        model_name="ResNet50"
    )
    
    # Log some training parameters
    tracker.log_params({
        "learning_rate": 0.001,
        "batch_size": 16,
        "optimizer": "Adam",
        "image_size": 224
    })
    
    # Simulate 3 rounds
    for round_num in range(1, 4):
        # Simulate metrics
        train_loss = 0.8 - (round_num * 0.05)
        eval_loss = 0.75 - (round_num * 0.05)
        accuracy = 0.65 + (round_num * 0.03)
        
        tracker.log_federated_round(
            round_num=round_num,
            train_loss=train_loss,
            eval_loss=eval_loss,
            accuracy=accuracy,
            num_clients=3
        )
    
    # End run
    tracker.end_run()
    
    print("\n✅ Demo completed! Check MLflow UI:")
    print("   Run: mlflow ui")
    print("   Then open: http://localhost:5000")


if __name__ == "__main__":
    demo_tracking()
