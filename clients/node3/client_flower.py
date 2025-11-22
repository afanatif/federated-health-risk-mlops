"""
Federated Learning Client for YOLOv8 Object Detection
Node 1 - Replace NODE_ID for other nodes
"""

import os
import torch
import flwr as fl
from ultralytics import YOLO

# Import model utilities
from models.model import get_model, model_to_ndarrays, ndarrays_to_model
from clients.common.image_dataset import convert_targets_to_yolov8_format

# Import node-specific loader
NODE_ID = 3  # <-- Change to 2, 3 for other nodes

try:
    loader_mod = __import__(f"clients.node{NODE_ID}.data_loader", fromlist=["get_loaders"])
    get_loaders = getattr(loader_mod, "get_loaders")
except Exception as e:
    raise RuntimeError(f"Could not import get_loaders for node{NODE_ID}: {e}")

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {DEVICE}")

# Load data
print(f"\n📁 Loading data for Node {NODE_ID}...")
train_loader, val_loader = get_loaders(batch_size=8, img_size=640)

# Create YOLOv8 model
print(f"\n🤖 Creating YOLOv8 model...")
model = get_model(model_size='n', num_classes=1, pretrained=False, img_size=640)
model.model.model.to(DEVICE)

print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")


class YOLOFLClient(fl.client.NumPyClient):
    """
    Flower client for YOLOv8 federated learning.
    """
    
    def __init__(self, model, train_loader, val_loader, node_id):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.node_id = node_id
        self.epochs_per_round = 1  # Train for 1 epoch per round
        
    def get_parameters(self, config=None):
        """Get model parameters as numpy arrays."""
        return model_to_ndarrays(self.model)
    
    def set_parameters(self, parameters, config=None):
        """Set model parameters from numpy arrays."""
        ndarrays_to_model(self.model, parameters)
    
    def fit(self, parameters, config=None):
        """
        Train YOLOv8 model on local data.
        
        Args:
            parameters: Global model parameters
            config: Optional training configuration
            
        Returns:
            Updated parameters, number of examples, metrics
        """
        print(f"\n{'='*60}")
        print(f"🏋️  Node {self.node_id}: Starting training round")
        print(f"{'='*60}")
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Train using ultralytics YOLO trainer
        # Save current model temporarily
        temp_model_path = f"temp_node{self.node_id}.pt"
        self.model.model.save(temp_model_path)
        
        # Create temporary dataset yaml for training
        data_yaml = f"""
path: data/federated/splits/iid_5nodes/node_{self.node_id}
train: images
val: images
nc: 1
names: ['pothole']
"""
        yaml_path = f"temp_node{self.node_id}.yaml"
        with open(yaml_path, 'w') as f:
            f.write(data_yaml)
        
        # Train model
        try:
            results = self.model.model.train(
                data=yaml_path,
                epochs=self.epochs_per_round,
                imgsz=640,
                batch=8,
                device=DEVICE,
                verbose=False,
                patience=0,  # No early stopping
                save=False,  # Don't save checkpoints
                plots=False,  # Don't create plots
                val=False    # Validate separately
            )
            
            # Get training loss
            train_loss = results.results_dict.get('train/box_loss', 0.0)
            
            print(f"✅ Node {self.node_id}: Training complete")
            print(f"   Train loss: {train_loss:.4f}")
            
        except Exception as e:
            print(f"⚠️  Training error: {e}")
            train_loss = 0.0
        finally:
            # Cleanup temporary files
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            if os.path.exists(yaml_path):
                os.remove(yaml_path)
        
        # Return updated parameters
        num_examples = len(self.train_loader.dataset)
        metrics = {"train_loss": float(train_loss)}
        
        return self.get_parameters(), num_examples, metrics
    
    def evaluate(self, parameters, config=None):
        """
        Evaluate YOLOv8 model on validation data.
        
        Args:
            parameters: Model parameters to evaluate
            config: Optional evaluation configuration
            
        Returns:
            Loss, number of examples, metrics
        """
        print(f"\n🔍 Node {self.node_id}: Evaluating model...")
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Put model in eval mode
        self.model.model.model.eval()
        
        total_loss = 0.0
        total_boxes = 0
        correct_detections = 0
        
        with torch.no_grad():
            for images, targets_list in self.val_loader:
                images = images.to(DEVICE)
                
                # Run inference
                try:
                    results = self.model.predict(images, conf=0.25, iou=0.45)
                    
                    # Count detections
                    for result in results:
                        if result.boxes is not None:
                            num_detections = len(result.boxes)
                            total_boxes += num_detections
                    
                    # For simplicity, use number of detections as metric
                    # In production, you'd calculate mAP, precision, recall, etc.
                    
                except Exception as e:
                    print(f"⚠️  Evaluation error: {e}")
                    continue
        
        # Calculate average metrics
        num_examples = len(self.val_loader.dataset)
        avg_boxes_per_image = total_boxes / num_examples if num_examples > 0 else 0.0
        
        # Use number of detections as a proxy for performance
        # Higher is generally better (but needs proper mAP in production)
        metrics = {
            "avg_detections": float(avg_boxes_per_image),
            "total_boxes": int(total_boxes)
        }
        
        print(f"✅ Node {self.node_id}: Evaluation complete")
        print(f"   Avg detections per image: {avg_boxes_per_image:.2f}")
        print(f"   Total boxes detected: {total_boxes}")
        
        # Return a simple loss metric (lower is better)
        # We use negative detections as "loss" so federated averaging works
        loss = -avg_boxes_per_image if avg_boxes_per_image > 0 else 1.0
        
        return float(loss), num_examples, metrics


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🚀 Starting Flower Client for Node {NODE_ID}")
    print(f"{'='*60}")
    
    # Create and start client
    client = YOLOFLClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        node_id=NODE_ID
    )
    
    # Connect to Flower server
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )
