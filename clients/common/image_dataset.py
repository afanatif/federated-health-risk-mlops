# clients/common/image_dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Configure here: if your label files use numeric class ids and pothole has index N, set below:
POTHOLE_CLASS_IDX = int(os.environ.get("POTHOLE_CLASS_IDX", "3"))  # default 3; change if needed
POTHOLE_CLASS_NAME = os.environ.get("POTHOLE_CLASS_NAME", "pothole").lower()

IMG_EXTS = (".jpg", ".jpeg", ".png")

DEFAULT_TRANSFORMS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class ImagePotholeDataset(Dataset):
    """
    Reads images and label txt files. Label formats supported:
      - Single integer in file: '3' -> class id
      - YOLO lines: '2 0.5 0.5 0.2 0.1' -> class id plus box coords
      - Class name lines: 'pothole' or lines with class names or single token
    Output:
      image tensor, binary label tensor float32 (1.0 if pothole present, else 0.0)
    """
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images dir not found: {images_dir}")
        if not os.path.isdir(labels_dir):
            raise FileNotFoundError(f"Labels dir not found: {labels_dir}")
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)])
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {images_dir}")
        self.transform = transform or DEFAULT_TRANSFORMS

    def __len__(self):
        return len(self.image_files)

    def _parse_label_file(self, label_path):
        """
        Return True if pothole present, else False.
        """
        if not os.path.exists(label_path):
            return False
        with open(label_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                # If first token is int-like -> class id
                first = parts[0]
                # try numeric
                try:
                    cls = int(float(first))
                    if cls == POTHOLE_CLASS_IDX:
                        return True
                    # else continue checking other lines
                except Exception:
                    # not numeric -> check if it's a known name
                    token = first.lower()
                    if token == POTHOLE_CLASS_NAME:
                        return True
                    # else maybe the whole line contains name tokens
                    if POTHOLE_CLASS_NAME in token:
                        return True
        return False

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        base = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.labels_dir, base + ".txt")
        has_pothole = self._parse_label_file(label_path)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(float(has_pothole), dtype=torch.float32)
