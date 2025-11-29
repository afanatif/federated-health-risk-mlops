"""
Model Manager for Federated Learning
Handles model versioning, registry, and lifecycle management with MLflow
"""
import mlflow
from mlflow.tracking import MlflowClient
import torch
from typing import Optional, Dict, List, Tuple
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage model versions and lifecycle with MLflow Model Registry."""
    
    def __init__(self, tracking_uri: str = "file:./mlruns", registered_model_name: str = "fl-global-model"):
        """
        Initialize model manager.
        
        Args:
            tracking_uri: MLflow tracking server URI
            registered_model_name: Name in MLflow Model Registry
        """
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.registered_model_name = registered_model_name
        
        # Create registered model if it doesn't exist
        try:
            self.client.get_registered_model(registered_model_name)
            logger.info(f"✅ Using existing model: {registered_model_name}")
        except mlflow.exceptions.RestException:
            mlflow.register_model(
                model_uri="models:/dummy",  # Will be replaced
                name=registered_model_name
            )
            logger.info(f"📦 Created new registered model: {registered_model_name}")
    
    def save_model_version(
        self,
        model_path: str,
        run_id: str,
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Save a new model version to the registry.
        
        Args:
            model_path: Path to saved model file
            run_id: MLflow run ID
            metrics: Model metrics
            tags: Additional tags
            
        Returns:
            Model version number
        """
        # Log model as artifact
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(model_path, "model")
            artifact_uri = mlflow.get_artifact_uri("model")
        
        # Register model version
        model_uri = f"{artifact_uri}/{os.path.basename(model_path)}"
        mv = self.client.create_model_version(
            name=self.registered_model_name,
            source=model_uri,
            run_id=run_id,
            tags=tags or {}
        )
        
        version = mv.version
        logger.info(f"📦 Saved model version {version}")
        
        return version
    
    def get_latest_version(self, stage: str = "None") -> Optional[str]:
        """
        Get the latest model version for a given stage.
        
        Args:
            stage: Model stage (None, Staging, Production, Archived)
            
        Returns:
            Model version number or None
        """
        try:
            versions = self.client.get_latest_versions(
                self.registered_model_name,
                stages=[stage] if stage != "None" else None
            )
            if versions:
                return versions[0].version
            return None
        except Exception as e:
            logger.error(f"Failed to get latest version: {e}")
            return None
    
    def promote_to_production(self, version: str):
        """
        Promote a model version to production.
        
        Args:
            version: Model version to promote
        """
        try:
            # Archive current production models
            for mv in self.client.search_model_versions(f"name='{self.registered_model_name}'"):
                if mv.current_stage == "Production":
                    self.client.transition_model_version_stage(
                        name=self.registered_model_name,
                        version=mv.version,
                        stage="Archived"
                    )
            
            # Promote new version
            self.client.transition_model_version_stage(
                name=self.registered_model_name,
                version=version,
                stage="Production"
            )
            logger.info(f"✅ Model v{version} promoted to Production")
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
    
    def promote_to_staging(self, version: str):
        """
        Promote a model version to staging.
        
        Args:
            version: Model version to promote
        """
        try:
            self.client.transition_model_version_stage(
                name=self.registered_model_name,
                version=version,
                stage="Staging"
            )
            logger.info(f"✅ Model v{version} promoted to Staging")
        except Exception as e:
            logger.error(f"Failed to promote to staging: {e}")
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """
        Compare two model versions.
        
        Args:
            version1: First version
            version2: Second version
            
        Returns:
            Comparison dictionary
        """
        try:
            mv1 = self.client.get_model_version(self.registered_model_name, version1)
            mv2 = self.client.get_model_version(self.registered_model_name, version2)
            
            # Get metrics from runs
            run1 = self.client.get_run(mv1.run_id)
            run2 = self.client.get_run(mv2.run_id)
            
            comparison = {
                'version1': {
                    'version': version1,
                    'stage': mv1.current_stage,
                    'metrics': run1.data.metrics,
                    'created': mv1.creation_timestamp
                },
                'version2': {
                    'version': version2,
                    'stage': mv2.current_stage,
                    'metrics': run2.data.metrics,
                    'created': mv2.creation_timestamp
                }
            }
            
            return comparison
        except Exception as e:
            logger.error(f"Failed to compare versions: {e}")
            return {}
    
    def list_all_versions(self) -> List[Dict]:
        """
        List all model versions.
        
        Returns:
            List of version information
        """
        try:
            versions = self.client.search_model_versions(f"name='{self.registered_model_name}'")
            return [
                {
                    'version': mv.version,
                    'stage': mv.current_stage,
                    'created': mv.creation_timestamp,
                    'run_id': mv.run_id
                }
                for mv in versions
            ]
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []
    
    def load_production_model(self, model_class, device: str = 'cpu'):
        """
        Load the current production model.
        
        Args:
            model_class: Model class to instantiate
            device: Device to load model on
            
        Returns:
            Loaded model
        """
        try:
            version = self.get_latest_version(stage="Production")
            if version is None:
                logger.warning("No production model found")
                return None
            
            mv = self.client.get_model_version(self.registered_model_name, version)
            
            # Download and load model
            model_uri = mv.source
            local_path = mlflow.artifacts.download_artifacts(model_uri)
            
            # Load into model
            model = model_class()
            model.load_state_dict(torch.load(local_path, map_location=device))
            
            logger.info(f"✅ Loaded production model v{version}")
            return model
        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            return None
    
    def delete_version(self, version: str):
        """
        Delete a model version.
        
        Args:
            version: Version to delete
        """
        try:
            self.client.delete_model_version(
                name=self.registered_model_name,
                version=version
            )
            logger.info(f"🗑️  Deleted model version {version}")
        except Exception as e:
            logger.error(f"Failed to delete version: {e}")
