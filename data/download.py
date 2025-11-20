# data/download.py
import os
import zipfile
import gdown
import shutil

# Drive file IDs you provided
NODE_LINKS = {
    "node1": "1medndzHlkC8hfRANmeYyejdaqf64l1Wm",
    "node2": "19MPz1BLPDIiPWBBaBykzt1XTB-v0CfOO",
    "node3": "1Dansjc5jn2hNeCMs2iW7Wod80EmQVO5L"
}

BASE = "data"  # base where node folders will be created

def download_zip(node_name, file_id, dest_zip):
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {node_name} -> {dest_zip}")
    gdown.download(url, dest_zip, quiet=False)

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

def find_first_csv(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".csv"):
                return os.path.join(root, f)
    return None

def ensure_node(node_name, file_id):
    node_dir = os.path.join(BASE, node_name)
    os.makedirs(node_dir, exist_ok=True)
    zip_path = os.path.join(node_dir, f"{node_name}.zip")

    # download zip if not present
    if not os.path.exists(zip_path):
        download_zip(node_name, file_id, zip_path)
    else:
        print(f"Zip already exists: {zip_path}")

    # extract into a temp folder under node_dir/extracted/
    extracted_dir = os.path.join(node_dir, "extracted")
    if not os.path.exists(extracted_dir) or not os.listdir(extracted_dir):
        extract_zip(zip_path, extracted_dir)
    else:
        print(f"Already extracted: {extracted_dir}")

    # search for CSV inside extracted_dir
    csv_path = find_first_csv(extracted_dir)
    if csv_path:
        target_csv = os.path.join(node_dir, "sample.csv")
        if not os.path.exists(target_csv):
            print(f"Found CSV: {csv_path} -> moving to {target_csv}")
            shutil.move(csv_path, target_csv)
        else:
            print(f"Target CSV already exists: {target_csv}")
    else:
        print(f"Warning: No CSV found in {extracted_dir}. Please place a CSV at {os.path.join(node_dir,'sample.csv')} manually.")

    # Optionally, warn about images/labels presence
    images_dir = os.path.join(node_dir, "images")
    labels_dir = os.path.join(node_dir, "labels")
    if os.path.exists(images_dir):
        print(f"Images folder exists: {images_dir}")
    if os.path.exists(labels_dir):
        print(f"Labels folder exists: {labels_dir}")

    print(f"Node {node_name} ready at {node_dir}")

def ensure_all():
    print("=== Preparing nodes ===")
    for n, fid in NODE_LINKS.items():
        ensure_node(n, fid)
    print("=== All done ===")

if __name__ == "__main__":
    ensure_all()
