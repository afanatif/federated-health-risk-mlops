import os
import zipfile
import gdown

# Google Drive file links
NODE_LINKS = {
    "node1": "https://drive.google.com/uc?id=1medndzHlkC8hfRANmeYyejdaqf64l1Wm",
    "node2": "https://drive.google.com/uc?id=19MPz1BLPDIiPWBBaBykzt1XTB-v0CfOO",
    "node3": "https://drive.google.com/uc?id=1Dansjc5jn2iW7Wod80EmQVO5L"
}

# Output target folders INSIDE clients folder
TARGET_DIRS = {
    "node1": "clients/node1/data",
    "node2": "clients/node2/data",
    "node3": "clients/node3/data"
}

def download_and_extract(node_name, file_id, target_folder):
    print(f"\n=== Processing {node_name} ===")

    os.makedirs(target_folder, exist_ok=True)

    zip_path = f"{target_folder}/{node_name}.zip"

    # Step 1: Download zip file
    print(f"Downloading {node_name} data...")
    gdown.download(file_id, zip_path, quiet=False)

    # Step 2: Extract
    print(f"Extracting {node_name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_folder)

    # Step 3: Cleanup zip
    os.remove(zip_path)

    print(f"{node_name} ready in: {target_folder}")


def ensure_all_nodes_ready():
    print("==== Starting dataset download ====")

    for node, link in NODE_LINKS.items():
        target = TARGET_DIRS[node]
        download_and_extract(node, link, target)

    print("\n==== All nodes are ready! ====")


if __name__ == "__main__":
    ensure_all_nodes_ready()
