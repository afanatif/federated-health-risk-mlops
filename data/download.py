# data/download.py
import os
import zipfile
import gdown
import shutil

# Google Drive file IDs (you provided these earlier)
NODE_LINKS = {
    "node1": "1medndzHlkC8hfRANmeYyejdaqf64l1Wm",
    "node2": "19MPz1BLPDIiPWBBaBykzt1XTB-v0CfOO",
    "node3": "1Dansjc5jn2hNeCMs2iW7Wod80EmQVO5L"
}

# where to create folders
CLIENTS_DIR = "clients"

def download_and_extract(node_name, file_id):
    node_data_dir = os.path.join(CLIENTS_DIR, node_name, "data")
    os.makedirs(node_data_dir, exist_ok=True)
    zip_path = os.path.join(node_data_dir, f"{node_name}.zip")

    # download zip if missing
    if not os.path.exists(zip_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {node_name} from {url} -> {zip_path}")
        gdown.download(url, zip_path, quiet=False)
    else:
        print(f"{zip_path} already present")

    # extract
    extracted_dir = os.path.join(node_data_dir, "extracted")
    if not os.path.exists(extracted_dir) or not os.listdir(extracted_dir):
        print(f"Extracting {zip_path} -> {extracted_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extracted_dir)
    else:
        print(f"Already extracted to {extracted_dir}")

    # attempt to move images/labels if they are nested inside extracted directory
    # If zip already has images/ and labels/ at top, keep them; else search.
    images_target = os.path.join(node_data_dir, "images")
    labels_target = os.path.join(node_data_dir, "labels")

    # helper to find folder by name recursively
    def find_folder(root, name):
        for root_, dirs, _ in os.walk(root):
            if name in dirs:
                return os.path.join(root_, name)
        return None

    # find images folder
    found_images = find_folder(extracted_dir, "images")
    found_labels = find_folder(extracted_dir, "labels")

    # if images_target exists and non-empty, keep
    if os.path.exists(images_target) and os.listdir(images_target):
        print(f"Images already at {images_target}")
    else:
        if found_images:
            print(f"Moving {found_images} -> {images_target}")
            if os.path.exists(images_target):
                shutil.rmtree(images_target)
            shutil.move(found_images, images_target)
        else:
            # attempt to find any images inside extracted_dir and move them into images/
            imgs = []
            for root_, _, files in os.walk(extracted_dir):
                for f in files:
                    if f.lower().endswith((".jpg",".jpeg",".png")):
                        imgs.append(os.path.join(root_, f))
            if imgs:
                os.makedirs(images_target, exist_ok=True)
                for p in imgs:
                    shutil.move(p, os.path.join(images_target, os.path.basename(p)))
                print(f"Collected {len(imgs)} images into {images_target}")
            else:
                print(f"Warning: no images found inside {extracted_dir}")

    if os.path.exists(labels_target) and os.listdir(labels_target):
        print(f"Labels already at {labels_target}")
    else:
        if found_labels:
            print(f"Moving {found_labels} -> {labels_target}")
            if os.path.exists(labels_target):
                shutil.rmtree(labels_target)
            shutil.move(found_labels, labels_target)
        else:
            # move any *.txt files as labels if present
            txts = []
            for root_, _, files in os.walk(extracted_dir):
                for f in files:
                    if f.lower().endswith(".txt"):
                        txts.append(os.path.join(root_, f))
            if txts:
                os.makedirs(labels_target, exist_ok=True)
                for p in txts:
                    shutil.move(p, os.path.join(labels_target, os.path.basename(p)))
                print(f"Collected {len(txts)} label txts into {labels_target}")
            else:
                print(f"Warning: no label txt files found inside {extracted_dir}")

    # optionally remove extracted dir (keep it for debugging)
    # shutil.rmtree(extracted_dir)
    print(f"Node {node_name} ready: images->{images_target}, labels->{labels_target}")

def ensure_all_nodes():
    print("=== Ensure nodes ===")
    for node, fid in NODE_LINKS.items():
        download_and_extract(node, fid)
    print("=== All nodes prepared ===")

if __name__ == "__main__":
    ensure_all_nodes()
