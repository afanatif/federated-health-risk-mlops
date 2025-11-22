"""
Data Preprocessing and Augmentation Pipeline
Handles: normalization, augmentation, feature engineering
"""
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
import random
from typing import Tuple, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DataPreprocessor:
    """Handles data preprocessing for federated learning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_size = config.get('image_size', 224)
        self.normalize_mean = config.get('normalize_mean', [0.485, 0.456, 0.406])
        self.normalize_std = config.get('normalize_std', [0.229, 0.224, 0.225])
        
    def get_transforms(self, mode='train'):
        """
        Get preprocessing transforms for training or validation.
        
        Args:
            mode: 'train' for training transforms, 'val' for validation
        
        Returns:
            Composition of transforms
        """
        if mode == 'train':
            return self._get_train_transforms()
        else:
            return self._get_val_transforms()
    
    def _get_train_transforms(self):
        """Training transforms with augmentation."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.normalize_mean,
                std=self.normalize_std
            )
        ])
    
    def _get_val_transforms(self):
        """Validation transforms without augmentation."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.normalize_mean,
                std=self.normalize_std
            )
        ])


class AdvancedAugmentation:
    """Advanced augmentation using Albumentations library."""
    
    def __init__(self, image_size=224):
        self.image_size = image_size
        
    def get_training_augmentation(self):
        """
        Get advanced training augmentation pipeline.
        Includes geometric, color, and quality transformations.
        """
        return A.Compose([
            # Resize
            A.Resize(self.image_size, self.image_size),
            
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                p=0.5
            ),
            A.RandomRotate90(p=0.3),
            
            # Quality & Blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(p=1.0),
            ], p=0.3),
            
            # Noise
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.2),
            
            # Color transforms
            A.OneOf([
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=1.0
                ),
            ], p=0.5),
            
            # Weather effects (useful for pothole detection)
            A.OneOf([
                A.RandomRain(p=1.0),
                A.RandomFog(p=1.0),
                A.RandomSunFlare(p=1.0),
                A.RandomShadow(p=1.0),
            ], p=0.2),
            
            # Distortion
            A.OneOf([
                A.ElasticTransform(p=1.0),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(p=1.0),
            ], p=0.2),
            
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    def get_validation_augmentation(self):
        """Get validation pipeline (no augmentation)."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])


class FeatureEngineering:
    """Extract additional features from images for better predictions."""
    
    @staticmethod
    def extract_texture_features(image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features from image.
        Useful for pothole detection (rough vs smooth surfaces).
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Calculate texture metrics
        features = {
            'mean_intensity': float(np.mean(gray)),
            'std_intensity': float(np.std(gray)),
            'entropy': float(-np.sum(gray * np.log2(gray + 1e-10))),
            'contrast': float(gray.max() - gray.min()),
        }
        
        return features
    
    @staticmethod
    def extract_edge_features(image: np.ndarray) -> Dict[str, float]:
        """
        Extract edge-based features.
        Potholes have distinct edge patterns.
        """
        # Simple edge detection using gradients
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Compute gradients
        gy, gx = np.gradient(gray)
        edge_magnitude = np.sqrt(gx**2 + gy**2)
        
        features = {
            'edge_mean': float(np.mean(edge_magnitude)),
            'edge_std': float(np.std(edge_magnitude)),
            'edge_density': float(np.sum(edge_magnitude > np.mean(edge_magnitude)) / edge_magnitude.size),
        }
        
        return features
    
    @staticmethod
    def extract_color_features(image: np.ndarray) -> Dict[str, float]:
        """Extract color distribution features."""
        if len(image.shape) != 3:
            return {}
        
        features = {}
        for i, channel in enumerate(['red', 'green', 'blue']):
            channel_data = image[:, :, i]
            features[f'{channel}_mean'] = float(np.mean(channel_data))
            features[f'{channel}_std'] = float(np.std(channel_data))
        
        return features
    
    @classmethod
    def extract_all_features(cls, image: np.ndarray) -> Dict[str, float]:
        """Extract all engineered features."""
        features = {}
        features.update(cls.extract_texture_features(image))
        features.update(cls.extract_edge_features(image))
        features.update(cls.extract_color_features(image))
        return features


class DataNormalizer:
    """Normalize data for better model convergence."""
    
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, data: np.ndarray):
        """Fit normalizer on training data."""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        return (data - self.mean) / (self.std + 1e-8)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted.")
        
        return data * (self.std + 1e-8) + self.mean


def create_preprocessing_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create complete preprocessing pipeline.
    
    Args:
        config: Configuration dictionary with preprocessing parameters
    
    Returns:
        Dictionary containing all preprocessing components
    """
    return {
        'basic_preprocessor': DataPreprocessor(config),
        'advanced_augmentation': AdvancedAugmentation(config.get('image_size', 224)),
        'feature_engineering': FeatureEngineering(),
        'normalizer': DataNormalizer(),
        'config': config
    }


def demonstrate_augmentation(image_path: str, output_dir: str = "augmentation_examples"):
    """
    Demonstrate augmentation by creating multiple augmented versions.
    Useful for visualizing the augmentation pipeline.
    """
    import os
    from PIL import Image
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = Image.open(image_path)
    image_np = np.array(image)
    
    # Create augmenter
    augmenter = AdvancedAugmentation(224)
    aug_pipeline = augmenter.get_training_augmentation()
    
    print(f"Creating 10 augmented versions of {image_path}...")
    
    # Generate augmented versions
    for i in range(10):
        augmented = aug_pipeline(image=image_np)
        aug_image = augmented['image']
        
        # Convert tensor to PIL Image
        aug_image_np = aug_image.permute(1, 2, 0).numpy()
        aug_image_np = (aug_image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
        aug_image_np = (aug_image_np * 255).clip(0, 255).astype(np.uint8)
        
        aug_pil = Image.fromarray(aug_image_np)
        aug_pil.save(f"{output_dir}/augmented_{i+1}.jpg")
    
    print(f"✅ Saved 10 augmented images to {output_dir}/")


if __name__ == "__main__":
    print("Data Preprocessing & Augmentation Module")
    print("="*80)
    
    # Example configuration
    config = {
        'image_size': 224,
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],
    }
    
    # Create pipeline
    pipeline = create_preprocessing_pipeline(config)
    
    print("✅ Preprocessing pipeline created with:")
    print("   • Basic preprocessing & normalization")
    print("   • Advanced augmentation (geometric, color, weather)")
    print("   • Feature engineering (texture, edges, color)")
    print("   • Data normalization")
    
    print("\n" + "="*80)
    print("💡 Usage in training:")
    print("   train_transform = pipeline['basic_preprocessor'].get_transforms('train')")
    print("   val_transform = pipeline['basic_preprocessor'].get_transforms('val')")
    print("="*80)
