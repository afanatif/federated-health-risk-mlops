import torch
from torchvision import transforms
from pathlib import Path


class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.transform = None

    def load(self):
        # Lazy import to avoid slowing down app startup
        from efficientdet_pytorch import EfficientDet
        from efficientdet_pytorch.utils import BBoxTransform, ClipBoxes

        config = {
            "num_classes": 1,      # pothole
            "network": "efficientdet-d0",
        }

        model = EfficientDet(config)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.to(self.device).eval()

        self.model = model
        self.bbox_transform = BBoxTransform()
        self.clip_boxes = ClipBoxes()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        return self

    def preprocess(self, image):
        """Convert PIL image → tensor"""
        return self.transform(image).unsqueeze(0).to(self.device)
