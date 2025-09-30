import json
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO)

DATASET_JSON = "bundles/wholeBody_ct_segmentation/configs/dataset_0.json"
SCRIPT = "data/pre_maskseg.py"

def is_dataset_empty(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        # works if JSON has {"training": [...]} or {"testing": [...]} structure
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 0:
                return False
        return True
    except Exception as e:
        logging.error(f"Error reading {path}: {e}")
        return True

def main():
    while True:
        if is_dataset_empty(DATASET_JSON):
            logging.info("✅ All cases processed, dataset_0.json is empty.")
            break

        logging.info("▶ Running segmentation step...")
        result = subprocess.run(["python", SCRIPT])
        
        if result.returncode != 0:
            logging.warning("⚠ Segmentation failed, retrying in 10s...")
            time.sleep(2)
        else:
            logging.info("✔ Step completed, checking for remaining cases...")

if __name__ == "__main__":
    main()
