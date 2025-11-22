from service.inference import run_inference

def test_yolov8_inference():
    image = "/home/dell/Assign/MLOPs/mlops_projects/training/data/raw/Pothole.v1i.yolov8/train/images/recorded_20240325_104726_0210_jpg.rf.3489628c77797a304186afc632a54c3b.jpg"  
    model = "models/yolov8/yolov8n.pt"

    result = run_inference(
        image_path=image,
        model_type="yolov8",
        model_path=model
    )

    print(result)
    assert result is not None
