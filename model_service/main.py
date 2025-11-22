#!/usr/bin/env python3
"""
main.py

Lightweight runner for the Model-Service-Repo.
Iterates images in a folder, calls service.inference.run_inference
for each image, displays annotated results and prints perf stats.

Usage:
  python main.py --images path/to/images --model-path models/yolov8/yolov8n.pt --max-frames 15
"""

import argparse
import time
import tempfile
import os
from pathlib import Path
import cv2
import base64
import traceback

from service.schemas import InferenceRequest
from service.inference import run_inference


def parse_args():
    p = argparse.ArgumentParser(description="Run batch inference using Model-Service-Repo")
    p.add_argument("--images", type=str, required=True,
                   help="Path to folder containing images to run inference on")
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to model file (e.g. models/yolov8/yolov8n.pt)")
    p.add_argument("--max-frames", type=int, default=15,
                   help="Max number of images to process for quick visual checks")
    p.add_argument("--frame-skip", type=int, default=1,
                   help="Process 1 every N images (1 = process all, 2 = every 2nd image)")
    p.add_argument("--downscale", type=float, default=1.0,
                   help="Downscale factor to apply before inference (e.g. 0.5). If 1.0 no change.")
    p.add_argument("--display", action="store_true", help="Show images in an OpenCV window")
    return p.parse_args()


def make_tmp_resized_copy(orig_path: Path, scale: float) -> Path:
    img = cv2.imread(str(orig_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {orig_path}")
    if scale == 1.0:
        return orig_path

    h, w = img.shape[:2]
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=orig_path.suffix)
    try:
        # close the fd returned by mkstemp to avoid leaking it
        os.close(tmp_fd)
        # write image to the temp path
        cv2.imwrite(tmp_path, resized)
    except Exception:
        # cleanup if any failure
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return Path(tmp_path)


def gather_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

import numpy as np

def _draw_manual_on_image(img, detections):
    """
    Draw detections on an OpenCV BGR image.
    `detections` can be:
      - list of dicts with keys: xmin/x1, ymin/y1, xmax/x2, ymax/y2, confidence/score, class/name
      - list of pydantic Box objects (attributes x1,y1,x2,y2,score,label)
      - ultralytics Boxes iterable (each box has .xyxy, .conf, .cls)
      - a Results object (we won't get here if plot/render worked)
    """
    for det in detections:
        # dict-like
        if isinstance(det, dict):
            # support both naming conventions
            x1 = det.get("x1") or det.get("xmin")
            y1 = det.get("y1") or det.get("ymin")
            x2 = det.get("x2") or det.get("xmax")
            y2 = det.get("y2") or det.get("ymax")
            conf = det.get("score") or det.get("confidence") or 0.0
            cls = det.get("label") or det.get("class") or det.get("name") or ""
        # pydantic Box (object with attributes)
        elif hasattr(det, "x1") and hasattr(det, "y1"):
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
            conf = getattr(det, "score", 0.0)
            cls = getattr(det, "label", "")
        # ultralytics Box-like (object with xyxy, conf, cls)
        elif hasattr(det, "xyxy") and hasattr(det, "conf"):
            try:
                coords = det.xyxy.squeeze().tolist()
            except Exception:
                coords = det.xyxy[0].tolist() if hasattr(det.xyxy, "__len__") else list(det.xyxy)
            x1, y1, x2, y2 = coords
            conf = float(det.conf.item()) if hasattr(det.conf, "item") else float(det.conf[0])
            # get class name if available via parent Results
            cls = getattr(det, "cls", "")
        # fallback: skip
        else:
            continue

        # convert to ints and clamp
        try:
            xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])
        except Exception:
            continue

        # draw
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"{cls} {conf:.2f}" if cls is not None else f"{conf:.2f}"
        cv2.putText(img, label, (max(xmin, 0), max(ymin - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def try_display_result(result, orig_path: Path):
    """
    Robust display helper for different result types.
    Returns True if an annotated image was shown.
    """
    # 1) If result is an ultralytics Results object (YOLOv8)
    try:
        # ultralytics Results implements .plot()
        if hasattr(result, "plot") and callable(result.plot):
            # result.plot() may return a numpy image (BGR) or list; handle both
            try:
                img_ann = result.plot()  # preferred for YOLOv8
            except Exception:
                img_ann = None

            if isinstance(img_ann, np.ndarray):
                cv2.imshow("inference", img_ann)
                return True
            # Sometimes result.plot() returns list-like
            if isinstance(img_ann, (list, tuple)) and len(img_ann):
                img0 = img_ann[0]
                if isinstance(img0, np.ndarray):
                    cv2.imshow("inference", img0)
                    return True
    except Exception:
        pass

    # 2) If result looks like YOLOv5 torch.hub results (has .render or .imgs)
    try:
        # results.render() draws on results.imgs in-place and returns list of images (BGR)
        if hasattr(result, "render") and callable(result.render):
            try:
                rendered = result.render()  # modifies result.imgs and returns list of np arrays
            except Exception:
                # some versions require calling result.render() without capture
                result.render()
                rendered = getattr(result, "imgs", None)
            if isinstance(rendered, list) and len(rendered) and isinstance(rendered[0], np.ndarray):
                cv2.imshow("inference", rendered[0])
                return True
            if getattr(result, "imgs", None):
                first = result.imgs[0]
                if isinstance(first, np.ndarray):
                    cv2.imshow("inference", first)
                    return True
    except Exception:
        pass

    # 3) If result is a mapping/list of boxes (your wrapper)
    try:
        # if result is pydantic InferenceResponse or list of Box objects/dicts
        detections = None
        # If pydantic model
        if hasattr(result, "detections"):
            detections = result.detections
        elif isinstance(result, (list, tuple)):
            detections = result
        elif isinstance(result, dict) and ("detections" in result):
            detections = result["detections"]

        if detections is not None:
            # read base image
            img = cv2.imread(str(orig_path))
            if img is None:
                return False

            # If detections is a YOLOv8 Results (iterable of Results), handle that:
            # e.g., some wrappers return [Results]
            if len(detections) and hasattr(detections[0], "boxes"):
                # treat as list of YOLO Results objects
                _results_list = detections
                for r in _results_list:
                    # r.boxes is iterable of Box objects
                    boxes = getattr(r, "boxes", [])
                    # boxes might be a Boxes object which is iterable
                    for b in boxes:
                        # build a tiny proxy det for manual drawer
                        try:
                            coords = b.xyxy.squeeze().tolist()
                        except Exception:
                            coords = b.xyxy[0].tolist() if hasattr(b.xyxy, "__len__") else list(b.xyxy)
                        det = {
                            "xmin": coords[0],
                            "ymin": coords[1],
                            "xmax": coords[2],
                            "ymax": coords[3],
                            "confidence": float(b.conf[0]) if hasattr(b, "conf") else 0.0,
                            "class": getattr(b, "cls", "")
                        }
                        img = _draw_manual_on_image(img, [det])
                cv2.imshow("inference", img)
                return True

            # Otherwise assume list-of-dicts or list-of-box-objects
            img = cv2.imread(str(orig_path))
            if img is None:
                return False
            img = _draw_manual_on_image(img, detections)
            cv2.imshow("inference", img)
            return True
    except Exception:
        pass

    # 4) Fallback: show original image
    try:
        img = cv2.imread(str(orig_path))
        if img is not None:
            cv2.imshow("inference", img)
            return True
    except Exception:
        pass

    return False


def main():
    args = parse_args()
    img_folder = Path(args.images)
    assert img_folder.exists() and img_folder.is_dir(), f"Images folder not found: {img_folder}"

    images = gather_images(img_folder)
    if len(images) == 0:
        print("No images found in", img_folder)
        return

    print(f"Found {len(images)} images — processing up to {args.max_frames} with skip={args.frame_skip}, downscale={args.downscale}")

    processed = 0
    t_total = 0.0
    tmp_files = []

    try:
        for idx, img_path in enumerate(images):
            if processed >= args.max_frames:
                break

            if (idx % args.frame_skip) != 0:
                continue

            input_path = img_path
            if args.downscale != 1.0:
                input_path = make_tmp_resized_copy(img_path, args.downscale)
                tmp_files.append(input_path)

            t0 = time.time()

            # prepare request
            try:
                img_bytes = input_path.read_bytes()
                img_b64 = base64.b64encode(img_bytes).decode()
                req = InferenceRequest(image_base64=img_b64)
            except Exception as e:
                print(f"Failed to prepare request for {input_path}: {e}")
                traceback.print_exc()
                continue

            # call inference and handle exceptions per-image so one failure doesn't stop the loop
            try:
                result = run_inference(req, model_path=str(args.model_path))
            except TypeError:
                # fallback if run_inference uses a different param name
                try:
                    result = run_inference(req)  # best-effort fallback
                except Exception as e:
                    print(f"Inference failed for {img_path.name}: {e}")
                    traceback.print_exc()
                    continue
            except Exception as e:
                print(f"Inference failed for {img_path.name}: {e}")
                traceback.print_exc()
                continue

            t1 = time.time()

            elapsed = (t1 - t0) * 1000.0
            t_total += (t1 - t0)
            processed += 1

            print(f"[{processed}] {img_path.name} — inference time: {elapsed:.1f} ms")

            displayed = False
            if args.display:
                try:
                    displayed = try_display_result(result, img_path)
                except Exception:
                    displayed = False

                if not displayed:
                    # fallback to original image display
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        cv2.imshow("inference", img)

                if cv2.waitKey(0) & 0xFF == ord('q'):
                    print("User requested exit")
                    break

    finally:
        for f in tmp_files:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
        cv2.destroyAllWindows()

    if processed:
        avg_ms = (t_total / processed) * 1000.0
        print(f"\nProcessed {processed} images — avg inference time: {avg_ms:.1f} ms")
    else:
        print("Processed 0 images")


if __name__ == "__main__":
    main()
