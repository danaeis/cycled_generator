import os
import json
import logging
import subprocess

from configs import ORIGINAL_DIR, SEGMENTATION_DIR, ORIGINAL_fixed_DIR

MONAI_DATA_DIR = ORIGINAL_fixed_DIR
# Recursively gather all NIfTI files (one per series)
nifti_paths = []
for study_uid in os.listdir(MONAI_DATA_DIR):
    study_path = os.path.join(MONAI_DATA_DIR, study_uid)
    if not os.path.isdir(study_path):
        continue
    for series_file in os.listdir(study_path):
        if series_file.endswith(".nii.gz"):
            full_path = os.path.join(study_path, series_file)
            nifti_paths.append(full_path.split("/")[-2]+"/"+full_path.split("/")[-1])

# Create a mapping of series_id to original volume path for quick lookup
series_to_original_path = {}
for path in nifti_paths:
    series_id = path.split("/")[-1].replace(".nii.gz", "")
    series_to_original_path[series_id] = os.path.join(ORIGINAL_DIR, path)

# Create JSON entries with relative paths or full paths
dataset_config = {
    "testing": [{"image": path} for path in nifti_paths]
}

output_suffix = "_seg.nii.gz"  # customize based on your naming


filtered_testing = []
for item in dataset_config["testing"]:
    base_name = os.path.basename(item["image"]).replace(".nii.gz", "")
    seg_path = os.path.join(SEGMENTATION_DIR, base_name + output_suffix)

    if not os.path.exists(seg_path):
        filtered_testing.append(item)

# Replace datalist with filtered version
filtered_data = {"testing": filtered_testing}

print(f"Filtered from {len(dataset_config['testing'])} to {len(filtered_testing)} cases.")


# Save to dataset_0.json
dataset_json_path = os.path.join("bundles/wholeBody_ct_segmentation/configs/dataset_0.json")
os.makedirs(os.path.dirname(dataset_json_path), exist_ok=True)
with open(dataset_json_path, 'w') as f:
    json.dump(filtered_data, f, indent=4)

print(f"✓ Updated dataset_0.json with {len(filtered_testing)} image paths.")

# Step 3: Run segmentation inference
logging.info("Step 3: Running segmentation inference")
# logging.info(f"Running segmentation on device: {device}")
# NOTE: The model weights in model.pt are a raw state_dict, not wrapped in a dict with a 'model' key.
# The inference.yaml config is set to load the weights directly.
mos = "multi-organ-segmentation"
ts = "totalsegmentator"
model = ts

if model == mos:
    config_file = "bundles/multi_organ_segmentation/configs/inference.yaml"
elif model == ts:
    config_file = "bundles/wholeBody_ct_segmentation/configs/inference.json"

cmd = [
    "python", "-m", "monai.bundle", "run",
    "--config_file", config_file
]

try:
    logging.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True
    )
    logging.info("Segmentation completed successfully")
    logging.info("Command stdout:")
    print(result.stdout) # Print stdout directly to see MONAI progress/messages
    if result.stderr:
        logging.warning("Command stderr:")
        logging.warning(result.stderr) # Log stderr as warning
except subprocess.CalledProcessError as e:
    logging.error(f"Error running segmentation: {e}")
    if e.stdout:
        logging.error(f"Command stdout: {e.stdout}")
    if e.stderr:
        logging.error(f"Command stderr: {e.stderr}")
    # Re-raise the exception to stop the script if segmentation fails
    raise
except FileNotFoundError:
    logging.error(f"Error: python or monai.bundle command not found. Make sure your environment is set up correctly.")
    raise
except Exception as e:
    logging.error(f"An unexpected error occurred during segmentation inference: {e}")
    raise