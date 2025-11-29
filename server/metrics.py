"""
Prometheus Metrics Exporter for Federated Learning Server
Exposes training metrics for monitoring and alerting
"""
from prometheus_client import Counter, Gauge, Histogram, start_http_server, CollectorRegistry
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FederatedMetrics:
    """Prometheus metrics collector for federated learning."""
    
    def __init__(self, port: int = 8081):
        """
        Initialize Prometheus metrics.
        
        Args:
            port: Port to expose metrics endpoint
        """
        self.port = port
        self.registry = CollectorRegistry()
        
        # Training round metrics
        self.rounds_completed = Counter(
            'fl_rounds_completed_total',
            'Number of federated training rounds completed',
            registry=self.registry
        )
        
        self.rounds_failed = Counter(
            'fl_rounds_failed_total',
            'Number of federated training rounds that failed',
            registry=self.registry
        )
        
        # Loss metrics
        self.train_loss = Gauge(
            'fl_train_loss',
            'Current training loss',
            registry=self.registry
        )
        
        self.eval_loss = Gauge(
            'fl_eval_loss',
            'Current evaluation loss',
            registry=self.registry
        )
        
        # Accuracy metrics
        self.accuracy = Gauge(
            'fl_accuracy',
            'Current model accuracy',
            registry=self.registry
        )
        
        self.best_accuracy = Gauge(
            'fl_best_accuracy',
            'Best accuracy achieved',
            registry=self.registry
        )
        
        # Client metrics
        self.active_clients = Gauge(
            'fl_active_clients',
            'Number of active clients in current round',
            registry=self.registry
        )
        
        self.total_clients = Gauge(
            'fl_total_clients',
            'Total number of registered clients',
            registry=self.registry
        )
        
        # Training time metrics
        self.round_duration = Histogram(
            'fl_round_duration_seconds',
            'Time taken to complete a training round',
            buckets=[30, 60, 120, 300, 600, 1800, 3600],
            registry=self.registry
        )
        
        # Model size metrics
        self.model_parameters = Gauge(
            'fl_model_parameters',
            'Number of model parameters',
            registry=self.registry
        )
        
        # Data metrics
        self.samples_per_round = Gauge(
            'fl_samples_per_round',
            'Number of training samples in current round',
            registry=self.registry
        )
        
        self._server_started = False
    
    def start_server(self):
        """Start Prometheus metrics HTTP server."""
        if not self._server_started:
            try:
                start_http_server(self.port, registry=self.registry)
                self._server_started = True
                logger.info(f"📊 Prometheus metrics server started on port {self.port}")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
    
    def record_round_complete(
        self,
        train_loss: float,
        eval_loss: float,
        accuracy: float,
        num_clients: int,
        duration: float,
        num_samples: Optional[int] = None
    ):
        """
        Record metrics for a completed training round.
        
        Args:
            train_loss: Training loss
            eval_loss: Evaluation loss
            accuracy: Model accuracy
            num_clients: Number of participating clients
            duration: Round duration in seconds
            num_samples: Total training samples
        """
        self.rounds_completed.inc()
        self.train_loss.set(train_loss)
        self.eval_loss.set(eval_loss)
        self.accuracy.set(accuracy)
        self.active_clients.set(num_clients)
        self.round_duration.observe(duration)
        
        # Update best accuracy if improved
        if accuracy > self.best_accuracy._value._value:
            self.best_accuracy.set(accuracy)
        
        if num_samples is not None:
            self.samples_per_round.set(num_samples)
    
    def record_round_failed(self):
        """Record a failed training round."""
        self.rounds_failed.inc()
    
    def set_total_clients(self, count: int):
        """Set total number of registered clients."""
        self.total_clients.set(count)
    
    def set_model_size(self, num_parameters: int):
        """Set number of model parameters."""
        self.model_parameters.set(num_parameters)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get current metric values.
        
        Returns:
            Dictionary of current metrics
        """
        return {
            'train_loss': self.train_loss._value._value,
            'eval_loss': self.eval_loss._value._value,
            'accuracy': self.accuracy._value._value,
            'best_accuracy': self.best_accuracy._value._value,
            'active_clients': self.active_clients._value._value,
        }


# Global metrics instance
_metrics: Optional[FederatedMetrics] = None


def get_metrics(port: int = 8081) -> FederatedMetrics:
    """
    Get or create global metrics instance.
    
    Args:
        port: Port for metrics server
        
    Returns:
        FederatedMetrics instance
    """
    global _metrics
    if _metrics is None:
        _metrics = FederatedMetrics(port=port)
    return _metrics


def start_metrics_server(port: int = 8081):
    """
    Start Prometheus metrics server.
    
    Args:
        port: Port to expose metrics
    """
    metrics = get_metrics(port)
    metrics.start_server()
