"""
Data Drift Detection for Federated Learning
Monitors distribution changes across federated rounds and clients
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Detects data drift in federated learning using statistical tests
    """
    
    def __init__(self, 
                 drift_threshold: float = 0.05,
                 window_size: int = 3,
                 mlflow_enabled: bool = True):
        """
        Initialize drift detector
        
        Args:
            drift_threshold: P-value threshold for drift detection (default: 0.05)
            window_size: Number of rounds to compare (default: 3)
            mlflow_enabled: Whether to log to MLflow
        """
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.mlflow_enabled = mlflow_enabled
        
        # Historical data storage
        self.metrics_history = []  # List of dicts per round
        self.class_distribution_history = []  # List of class distributions
        
        logger.info(f"DataDriftDetector initialized (threshold={drift_threshold}, window={window_size})")
    
    def detect_metric_drift(self, 
                           current_metrics: Dict[str, float],
                           round_num: int) -> Dict[str, any]:
        """
        Detect drift in performance metrics using statistical tests
        
        Args:
            current_metrics: Current round metrics (loss, accuracy, etc.)
            round_num: Current training round
            
        Returns:
            Dictionary with drift detection results
        """
        self.metrics_history.append({
            'round': round_num,
            'metrics': current_metrics,
            'timestamp': datetime.now()
        })
        
        drift_results = {
            'round': round_num,
            'drift_detected': False,
            'drifted_metrics': [],
            'drift_scores': {}
        }
        
        # Need at least window_size rounds for comparison
        if len(self.metrics_history) < self.window_size + 1:
            logger.info(f"Round {round_num}: Insufficient history for drift detection ({len(self.metrics_history)}/{self.window_size + 1})")
            return drift_results
        
        # Compare current metrics with historical window
        historical_window = self.metrics_history[-(self.window_size + 1):-1]
        
        for metric_name in current_metrics.keys():
            if metric_name in ['error', 'num_examples']:
                continue  # Skip non-numeric or count metrics
            
            # Extract historical values
            historical_values = [
                h['metrics'].get(metric_name, 0) 
                for h in historical_window 
                if metric_name in h['metrics']
            ]
            
            if len(historical_values) < 2:
                continue
            
            current_value = current_metrics[metric_name]
            
            # Perform Kolmogorov-Smirnov test
            # Compare current value distribution with historical
            try:
                # Use z-score to detect outliers
                mean = np.mean(historical_values)
                std = np.std(historical_values)
                
                if std > 0:
                    z_score = abs((current_value - mean) / std)
                    
                    # Z-score > 2 indicates potential drift (95% confidence)
                    # Z-score > 3 indicates significant drift (99.7% confidence)
                    drift_detected = z_score > 2.0
                    
                    drift_results['drift_scores'][metric_name] = {
                        'z_score': float(z_score),
                        'current_value': float(current_value),
                        'historical_mean': float(mean),
                        'historical_std': float(std),
                        'drift_detected': drift_detected
                    }
                    
                    if drift_detected:
                        drift_results['drift_detected'] = True
                        drift_results['drifted_metrics'].append(metric_name)
                        logger.warning(
                            f"⚠️  DRIFT DETECTED in {metric_name}: "
                            f"current={current_value:.4f}, mean={mean:.4f}, "
                            f"z-score={z_score:.2f}"
                        )
            
            except Exception as e:
                logger.error(f"Error detecting drift for {metric_name}: {e}")
        
        # Log to MLflow if enabled
        if self.mlflow_enabled:
            self._log_drift_to_mlflow(drift_results, round_num)
        
        return drift_results
    
    def detect_class_distribution_drift(self,
                                       class_counts: Dict[str, int],
                                       round_num: int) -> Dict[str, any]:
        """
        Detect drift in class distribution using Chi-Square test
        
        Args:
            class_counts: Dictionary of class names to sample counts
            round_num: Current training round
            
        Returns:
            Dictionary with drift detection results
        """
        self.class_distribution_history.append({
            'round': round_num,
            'distribution': class_counts,
            'timestamp': datetime.now()
        })
        
        drift_results = {
            'round': round_num,
            'distribution_drift_detected': False,
            'chi_square_statistic': None,
            'p_value': None
        }
        
        # Need at least 2 rounds for comparison
        if len(self.class_distribution_history) < 2:
            return drift_results
        
        try:
            # Get previous distribution
            prev_dist = self.class_distribution_history[-2]['distribution']
            
            # Ensure same classes
            all_classes = set(class_counts.keys()) | set(prev_dist.keys())
            
            current_counts = np.array([class_counts.get(c, 0) for c in sorted(all_classes)])
            previous_counts = np.array([prev_dist.get(c, 0) for c in sorted(all_classes)])
            
            # Perform Chi-Square test
            if current_counts.sum() > 0 and previous_counts.sum() > 0:
                chi2_stat, p_value = stats.chisquare(
                    current_counts + 1,  # Add 1 to avoid zeros
                    previous_counts + 1
                )
                
                drift_detected = p_value < self.drift_threshold
                
                drift_results.update({
                    'distribution_drift_detected': drift_detected,
                    'chi_square_statistic': float(chi2_stat),
                    'p_value': float(p_value)
                })
                
                if drift_detected:
                    logger.warning(
                        f"⚠️  CLASS DISTRIBUTION DRIFT DETECTED: "
                        f"chi2={chi2_stat:.2f}, p-value={p_value:.4f}"
                    )
                else:
                    logger.info(
                        f"✓ No class distribution drift: p-value={p_value:.4f}"
                    )
        
        except Exception as e:
            logger.error(f"Error detecting class distribution drift: {e}")
        
        return drift_results
    
    def detect_client_drift(self,
                           client_metrics: List[Dict[str, float]],
                           round_num: int) -> Dict[str, any]:
        """
        Detect drift between clients (heterogeneity detection)
        
        Args:
            client_metrics: List of metric dictionaries from each client
            round_num: Current training round
            
        Returns:
            Dictionary with client drift results
        """
        drift_results = {
            'round': round_num,
            'client_heterogeneity_detected': False,
            'heterogeneous_metrics': []
        }
        
        if len(client_metrics) < 2:
            return drift_results
        
        # Check variance across clients for each metric
        metric_names = set()
        for cm in client_metrics:
            metric_names.update(cm.keys())
        
        for metric_name in metric_names:
            if metric_name in ['error', 'num_examples']:
                continue
            
            values = [
                cm.get(metric_name, 0) 
                for cm in client_metrics 
                if metric_name in cm
            ]
            
            if len(values) < 2:
                continue
            
            # Calculate coefficient of variation (CV)
            mean = np.mean(values)
            std = np.std(values)
            
            if mean > 0:
                cv = std / mean
                
                # CV > 0.3 indicates high heterogeneity
                if cv > 0.3:
                    drift_results['client_heterogeneity_detected'] = True
                    drift_results['heterogeneous_metrics'].append({
                        'metric': metric_name,
                        'coefficient_of_variation': float(cv),
                        'mean': float(mean),
                        'std': float(std),
                        'values': [float(v) for v in values]
                    })
                    
                    logger.warning(
                        f"⚠️  HIGH CLIENT HETEROGENEITY in {metric_name}: "
                        f"CV={cv:.2f}, values={values}"
                    )
        
        return drift_results
    
    def _log_drift_to_mlflow(self, drift_results: Dict, round_num: int):
        """Log drift detection results to MLflow"""
        try:
            import mlflow
            
            # Log drift flag
            mlflow.log_metric("drift_detected", 1.0 if drift_results['drift_detected'] else 0.0, step=round_num)
            
            # Log drift scores
            for metric_name, scores in drift_results.get('drift_scores', {}).items():
                mlflow.log_metric(f"drift_zscore_{metric_name}", scores['z_score'], step=round_num)
            
            # Log number of drifted metrics
            mlflow.log_metric("num_drifted_metrics", len(drift_results['drifted_metrics']), step=round_num)
            
        except Exception as e:
            logger.error(f"Failed to log drift to MLflow: {e}")
    
    def get_drift_summary(self) -> Dict:
        """Get summary of all detected drifts"""
        total_rounds = len(self.metrics_history)
        
        drift_count = sum(
            1 for h in self.metrics_history 
            if h.get('drift_detected', False)
        )
        
        return {
            'total_rounds': total_rounds,
            'rounds_with_drift': drift_count,
            'drift_rate': drift_count / total_rounds if total_rounds > 0 else 0,
            'current_round': self.metrics_history[-1]['round'] if self.metrics_history else 0
        }


def create_drift_detector(config: Optional[Dict] = None) -> DataDriftDetector:
    """
    Factory function to create drift detector with configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured DataDriftDetector instance
    """
    if config is None:
        config = {}
    
    return DataDriftDetector(
        drift_threshold=config.get('drift_threshold', 0.05),
        window_size=config.get('window_size', 3),
        mlflow_enabled=config.get('mlflow_enabled', True)
    )
