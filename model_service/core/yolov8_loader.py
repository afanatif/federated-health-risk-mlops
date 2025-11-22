from pathlib import Path
from ultralytics import YOLO

class YOLOv8Loader:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None

    def load(self):
        self.model = YOLO(str(self.model_path))
        return self

    def predict(self, image_path: str):
        result = self.model(image_path)
        return result[0]  # first prediction
