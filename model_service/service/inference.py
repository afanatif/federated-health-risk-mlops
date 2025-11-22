import time
import base64
from pathlib import Path
from typing import List, Optional
from io import BytesIO

import torch
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from service.schemas import InferenceRequest, InferenceResponse, Box


# ---------------------------------------------------
# Model Cache
# ---------------------------------------------------
MODEL_DIR = "model_service/models"
_LOADED_MODELS = {}

def load_model(model_path: str) -> YOLO:
    """Load YOLO model from absolute path or from MODEL_DIR."""

    # Convert to Path
    mp = Path(model_path)

    # If user gave just a filename, assume it's inside MODEL_DIR
    if not mp.exists():
        mp = MODEL_DIR / model_path

    if not mp.exists():
        raise FileNotFoundError(f"Model file not found: {mp}")

    # Use cache
    key = str(mp)
    if key not in _LOADED_MODELS:
        _LOADED_MODELS[key] = YOLO(str(mp))

    return _LOADED_MODELS[key]



# ---------------------------------------------------
# Base64 → PIL Image
# ---------------------------------------------------
def decode_image(req: InferenceRequest) -> Image.Image:
    img_bytes = base64.b64decode(req.image_base64)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return img


# ---------------------------------------------------
# Inference
# ---------------------------------------------------
def run_inference(
    req: InferenceRequest,
    model_path: str = "yolov8l.pt",
    imgsz: int = 1024,
    conf: float = 0.1,
    iou: float = 0.45
) -> InferenceResponse:

    # Load the model using the provided `model_path`
    model = load_model(model_path)
    img = decode_image(req)

    start = time.time()
    results = model.predict(
        source=img,
        imgsz=imgsz,
        conf=0.1,
        iou=iou,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )
    end = time.time()

    detections = []
    for r in results:
        boxes = r.boxes  # YOLOv8 Boxes object

        for box in boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            # print(box.xyxy, box.conf, box.cls)

            detections.append({
                "x1": float(xmin),
                "y1": float(ymin),
                "x2": float(xmax),
                "y2": float(ymax),
                "score": float(box.conf[0]),
                "class": model.names[int(box.cls[0])]
            })

    return InferenceResponse(
        detections=detections,
        inference_time_ms=(end - start) * 1000
    )




# ---------------------------------------------------
# Quick CLI test
# ---------------------------------------------------
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 3:
        print("Usage: python inference.py <image_path> <model_path>")
        exit()

    img_path = Path(sys.argv[1])
    model_path = sys.argv[2]

    img_bytes = img_path.read_bytes()
    img_b64 = base64.b64encode(img_bytes).decode()

    req = InferenceRequest(image_base64=img_b64)
    resp = run_inference(req, model_path=model_path)
    print(json.dumps(resp.model_dump(), indent=4))
