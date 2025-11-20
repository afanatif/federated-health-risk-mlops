import os
import gdown
import zipfile

DATA_DIR = "data"

# Google Drive file IDs extracted from your links
NODE_LINKS = {
    "node1": "1medndzHlkC8hfRANmeYyejdaqf64l1Wm",
    "node2": "19MPz1BLPDIiPWBBaBykzt1XTB-v0CfOO",
    "node3": "1Dansjc5jn2hNeCMs2iW7Wod80EmQVO5L"
}

def download_file(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading: {dest_path}")
    gdown.download(url, dest_path, quiet=False)


def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} → {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def ensure_node(node_name, file_id):
    node_zip = f"{DATA_DIR}/{node_name}.zip"
    node_folder = f"{DATA_DIR}/{node_name}"

    # Create data folder if missing
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Download zip if missing
    if not os.path.exists(node_zip):
        download_file(file_id, node_zip)
    else:
        print(f"✔ {node_zip} already exists")

    # 2. Extract zip if not extracted
    if not os.path.exists(node_folder):
        extract_zip(node_zip, node_folder)
    else:
        print(f"✔ {node_folder} already extracted")

    # 3. Verify required subfolders
    images_path = os.path.join(node_folder, "images")
    labels_path = os.path.join(node_folder, "labels")

    if not os.path.exists(images_path):
        print(f"❌ Warning: {images_path} missing!")
    if not os.path.exists(labels_path):
        print(f"❌ Warning: {labels_path} missing!")

    print(f"✔ Node {node_name} ready.\n")


def ensure_data_ready():
    print("\n=== Checking Dataset for All Nodes ===\n")
    for node_name, file_id in NODE_LINKS.items():
        ensure_node(node_name, file_id)
    print("\n=== All nodes successfully prepared ===\n")


if __name__ == "__main__":
    ensure_data_ready()
