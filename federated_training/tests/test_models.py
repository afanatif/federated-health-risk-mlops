"""
Comprehensive Unit Tests for Federated Learning System
Tests: models, data loaders, preprocessing, federated operations
"""
import unittest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestModelArchitecture(unittest.TestCase):
    """Test model creation and forward pass."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from models.model import get_model
            self.model = get_model(pretrained=False, backbone="resnet18")  # Use ResNet18 for faster tests
        except ImportError:
            self.skipTest("Model module not found")
    
    def test_model_creation(self):
        """Test that model can be created."""
        self.assertIsNotNone(self.model)
        self.assertTrue(hasattr(self.model, 'forward'))
    
    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = self.model(dummy_input)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 1))
        
        # Check output range (sigmoid should be 0-1)
        self.assertTrue((output >= 0).all())
        self.assertTrue((output <= 1).all())
    
    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        params = list(self.model.parameters())
        self.assertGreater(len(params), 0)
        
        # Check that parameters require gradients
        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        self.assertGreater(trainable_params, 0)
    
    def test_batch_sizes(self):
        """Test model works with different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            dummy_input = torch.randn(batch_size, 3, 224, 224)
            with torch.no_grad():
                output = self.model(dummy_input)
            self.assertEqual(output.shape[0], batch_size)


class TestModelConversion(unittest.TestCase):
    """Test model parameter conversion utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from models.model import get_model, model_to_ndarrays, ndarrays_to_model
            self.model = get_model(pretrained=False, backbone="resnet18")
            self.model_to_ndarrays = model_to_ndarrays
            self.ndarrays_to_model = ndarrays_to_model
        except ImportError:
            self.skipTest("Model module not found")
    
    def test_model_to_arrays(self):
        """Test converting model to numpy arrays."""
        arrays = self.model_to_ndarrays(self.model)
        
        # Check we got a list
        self.assertIsInstance(arrays, list)
        self.assertGreater(len(arrays), 0)
        
        # Check all elements are numpy arrays
        for arr in arrays:
            self.assertIsInstance(arr, np.ndarray)
    
    def test_arrays_to_model(self):
        """Test loading arrays back into model."""
        # Get original arrays
        original_arrays = self.model_to_ndarrays(self.model)
        
        # Create new model
        from models.model import get_model
        new_model = get_model(pretrained=False, backbone="resnet18")
        
        # Load arrays
        self.ndarrays_to_model(new_model, original_arrays)
        
        # Get new arrays
        new_arrays = self.model_to_ndarrays(new_model)
        
        # Check they match
        for orig, new in zip(original_arrays, new_arrays):
            np.testing.assert_array_equal(orig, new)
    
    def test_conversion_preserves_predictions(self):
        """Test that conversion preserves model predictions."""
        # Create dummy input
        dummy_input = torch.randn(2, 3, 224, 224)
        
        # Get original prediction
        with torch.no_grad():
            original_pred = self.model(dummy_input)
        
        # Convert and reload
        arrays = self.model_to_ndarrays(self.model)
        from models.model import get_model
        new_model = get_model(pretrained=False, backbone="resnet18")
        self.ndarrays_to_model(new_model, arrays)
        
        # Get new prediction
        with torch.no_grad():
            new_pred = new_model(dummy_input)
        
        # Check predictions match
        torch.testing.assert_close(original_pred, new_pred)


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing and augmentation."""
    
    def test_basic_transforms(self):
        """Test basic image transforms."""
        try:
            from scripts.data_preprocessing import DataPreprocessor
        except ImportError:
            self.skipTest("Preprocessing module not found")
        
        config = {
            'image_size': 224,
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225]
        }
        
        preprocessor = DataPreprocessor(config)
        
        # Get transforms
        train_transform = preprocessor.get_transforms('train')
        val_transform = preprocessor.get_transforms('val')
        
        self.assertIsNotNone(train_transform)
        self.assertIsNotNone(val_transform)
    
    def test_normalization(self):
        """Test data normalization."""
        try:
            from scripts.data_preprocessing import DataNormalizer
        except ImportError:
            self.skipTest("Preprocessing module not found")
        
        # Create dummy data
        data = np.random.randn(100, 10)
        
        normalizer = DataNormalizer()
        normalized = normalizer.fit_transform(data)
        
        # Check normalized data has mean ~0 and std ~1
        self.assertAlmostEqual(np.mean(normalized), 0, places=1)
        self.assertAlmostEqual(np.std(normalized), 1, places=1)
        
        # Check inverse transform
        denormalized = normalizer.inverse_transform(normalized)
        np.testing.assert_array_almost_equal(data, denormalized, decimal=5)


class TestDataValidation(unittest.TestCase):
    """Test data validation functionality."""
    
    def test_validator_creation(self):
        """Test that validator can be created."""
        try:
            from scripts.data_validation import DataValidator
        except ImportError:
            self.skipTest("Validation module not found")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = DataValidator(tmpdir)
            self.assertIsNotNone(validator)


class TestFederatedOperations(unittest.TestCase):
    """Test federated learning operations."""
    
    def test_weighted_average(self):
        """Test weighted average function."""
        try:
            from server.server_flower import weighted_average
        except ImportError:
            self.skipTest("Server module not found")
        
        # Create test metrics
        metrics = [
            (100, {"accuracy": 0.8, "loss": 0.3}),
            (200, {"accuracy": 0.7, "loss": 0.4}),
            (150, {"accuracy": 0.75, "loss": 0.35})
        ]
        
        result = weighted_average(metrics)
        
        # Check result structure
        self.assertIn("accuracy", result)
        self.assertIn("loss", result)
        
        # Check weighted average calculation
        # Expected accuracy: (100*0.8 + 200*0.7 + 150*0.75) / 450 = 0.738888...
        expected_accuracy = (100*0.8 + 200*0.7 + 150*0.75) / 450
        self.assertAlmostEqual(result["accuracy"], expected_accuracy, places=5)
    
    def test_empty_metrics(self):
        """Test weighted average with empty metrics."""
        try:
            from server.server_flower import weighted_average
        except ImportError:
            self.skipTest("Server module not found")
        
        result = weighted_average([])
        self.assertEqual(result, {})


class TestExperimentTracking(unittest.TestCase):
    """Test MLflow experiment tracking."""
    
    def test_tracker_creation(self):
        """Test that tracker can be created."""
        try:
            from scripts.experiment_tracking import ExperimentTracker
        except ImportError:
            self.skipTest("Experiment tracking module not found")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="test_experiment",
                tracking_uri=f"file:{tmpdir}"
            )
            self.assertIsNotNone(tracker)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""
    
    def test_full_training_step(self):
        """Test a complete training step."""
        try:
            from models.model import get_model
        except ImportError:
            self.skipTest("Model module not found")
        
        # Create model
        model = get_model(pretrained=False, backbone="resnet18")
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create dummy batch
        images = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 2, (4,)).float()
        
        # Forward pass
        outputs = model(images).squeeze()
        
        # Calculate loss
        criterion = torch.nn.BCELoss()
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check loss is finite
        self.assertTrue(torch.isfinite(loss))
        
        # Check gradients exist
        has_gradients = any(
            p.grad is not None and torch.any(p.grad != 0)
            for p in model.parameters()
        )
        self.assertTrue(has_gradients)


def run_tests():
    """Run all tests and generate report."""
    print("="*80)
    print("RUNNING UNIT TESTS")
    print("="*80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestModelArchitecture))
    suite.addTests(loader.loadTestsFromTestCase(TestModelConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestFederatedOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestExperimentTracking))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")
    
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
