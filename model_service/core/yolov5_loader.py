import torch
from pathlib import Path

class YOLOv5Loader:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None

    def load(self):
        self.model = torch.hub.load(
            'ultralytics/yolov5', 
            'custom', 
            path=str(self.model_path), 
            source='github'
        )
        self.model.eval()
        return self

    def predict(self, image_path: str):
        result = self.model(image_path)
        return result
