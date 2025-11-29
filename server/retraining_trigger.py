"""
Automated Re-training Trigger for Federated Learning
Monitors performance and triggers re-training based on:
- Performance degradation
- Scheduled intervals
- Manual triggers
"""
import logging
import schedule
import time
from typing import Callable, Optional, Dict
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Re-training trigger types."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    DATA_DRIFT = "data_drift"


@dataclass
class RetrainingTrigger:
    """Re-training trigger event."""
    trigger_type: TriggerType
    timestamp: datetime
    reason: str
    metrics: Optional[Dict[str, float]] = None


class RetrainingManager:
    """Manage automated re-training triggers."""
    
    def __init__(
        self,
        accuracy_threshold: float = 0.05,  # 5% drop triggers retraining
        check_interval: int = 300,  # Check every 5 minutes
        min_rounds_between_retrain: int = 10
    ):
        """
        Initialize retraining manager.
        
        Args:
            accuracy_threshold: Accuracy drop threshold for triggering
            check_interval: Seconds between performance checks
            min_rounds_between_retrain: Minimum rounds before allowing retrain
        """
        self.accuracy_threshold = accuracy_threshold
        self.check_interval = check_interval
        self.min_rounds_between_retrain = min_rounds_between_retrain
        
        self.best_accuracy = 0.0
        self.current_accuracy = 0.0
        self.rounds_since_retrain = 0
        self.last_retrain_time = None
        
        self.retrain_callback: Optional[Callable] = None
        self.monitoring = False
        self.monitor_thread = None
        
        # History
        self.trigger_history = []
    
    def set_retrain_callback(self, callback: Callable):
        """
        Set callback function to execute when re-training is triggered.
        
        Args:
            callback: Function to call for re-training
        """
        self.retrain_callback = callback
        logger.info("✅ Re-training callback registered")
    
    def update_metrics(self, accuracy: float, round_num: int):
        """
        Update current metrics and check for degradation.
        
        Args:
            accuracy: Current model accuracy
            round_num: Current round number
        """
        self.current_accuracy = accuracy
        self.rounds_since_retrain = round_num
        
        # Update best accuracy
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            logger.info(f"📊 New best accuracy: {accuracy:.4f}")
        
        # Check for degradation
        self._check_performance_degradation()
    
    def _check_performance_degradation(self):
        """Check if performance has degraded beyond threshold."""
        if self.best_accuracy == 0:
            return
        
        degradation = self.best_accuracy - self.current_accuracy
        
        if degradation > self.accuracy_threshold:
            if self.rounds_since_retrain >= self.min_rounds_between_retrain:
                reason = f"Accuracy dropped by {degradation:.2%} (from {self.best_accuracy:.4f} to {self.current_accuracy:.4f})"
                self._trigger_retraining(
                    TriggerType.PERFORMANCE_DEGRADATION,
                    reason,
                    {'best_accuracy': self.best_accuracy, 'current_accuracy': self.current_accuracy}
                )
    
    def manual_trigger(self, reason: str = "Manual trigger"):
        """
        Manually trigger re-training.
        
        Args:
            reason: Reason for manual trigger
        """
        self._trigger_retraining(TriggerType.MANUAL, reason)
    
    def schedule_periodic_retrain(self, interval_days: int = 7):
        """
        Schedule periodic re-training.
        
        Args:
            interval_days: Days between scheduled re-training
        """
        def scheduled_retrain():
            reason = f"Scheduled re-training (every {interval_days} days)"
            self._trigger_retraining(TriggerType.SCHEDULED, reason)
        
        schedule.every(interval_days).days.do(scheduled_retrain)
        logger.info(f"📅 Scheduled re-training every {interval_days} days")
    
    def _trigger_retraining(
        self,
        trigger_type: TriggerType,
        reason: str,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Trigger re-training.
        
        Args:
            trigger_type: Type of trigger
            reason: Reason for triggering
            metrics: Optional metrics context
        """
        trigger = RetrainingTrigger(
            trigger_type=trigger_type,
            timestamp=datetime.now(),
            reason=reason,
            metrics=metrics
        )
        
        self.trigger_history.append(trigger)
        self.last_retrain_time = trigger.timestamp
        self.rounds_since_retrain = 0
        
        logger.warning(f"🔄 RE-TRAINING TRIGGERED: {reason}")
        
        # Execute callback if registered
        if self.retrain_callback:
            try:
                self.retrain_callback(trigger)
                logger.info("✅ Re-training callback executed")
            except Exception as e:
                logger.error(f"❌ Re-training callback failed: {e}")
        else:
            logger.warning("⚠️  No re-training callback registered")
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("👀 Re-training monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Re-training monitoring stopped")
    
    def get_trigger_history(self) -> list:
        """Get history of all triggers."""
        return [
            {
                'type': t.trigger_type.value,
                'timestamp': t.timestamp.isoformat(),
                'reason': t.reason,
                'metrics': t.metrics
            }
            for t in self.trigger_history
        ]
    
    def reset_metrics(self):
        """Reset tracking metrics."""
        self.best_accuracy = 0.0
        self.current_accuracy = 0.0
        self.rounds_since_retrain = 0
        logger.info("🔄 Metrics reset")


# REST API endpoint for manual triggering (Flask)
def create_trigger_api(retrain_manager: RetrainingManager):
    """
    Create Flask API for manual re-training triggers.
    
    Args:
        retrain_manager: RetrainingManager instance
        
    Returns:
        Flask app
    """
    try:
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        
        @app.route('/trigger_retrain', methods=['POST'])
        def trigger_retrain():
            """Trigger re-training via POST request."""
            data = request.get_json() or {}
            reason = data.get('reason', 'Manual API trigger')
            
            retrain_manager.manual_trigger(reason)
            
            return jsonify({
                'status': 'success',
                'message': 'Re-training triggered',
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }), 200
        
        @app.route('/trigger_history', methods=['GET'])
        def get_history():
            """Get re-training trigger history."""
            return jsonify({
                'history': retrain_manager.get_trigger_history()
            }), 200
        
        @app.route('/metrics', methods=['GET'])
        def get_metrics():
            """Get current metrics."""
            return jsonify({
                'best_accuracy': retrain_manager.best_accuracy,
                'current_accuracy': retrain_manager.current_accuracy,
                'rounds_since_retrain': retrain_manager.rounds_since_retrain,
                'last_retrain': retrain_manager.last_retrain_time.isoformat() if retrain_manager.last_retrain_time else None
            }), 200
        
        return app
    except ImportError:
        logger.warning("Flask not installed, API not available")
        return None
