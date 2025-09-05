"""
Requirements:
- Dataset Name and corresponding `dataset.json` file.
- Training configuration ('2D' or '3D') and fold setup.
- Preprocessing plan generated via the `plan_preprocess_commands` pipeline.
- Correct paths defined in the project environment.

Inputs:
- `dataset_name`: The name of the dataset to be trained on.

Outputs:
- Trained model and associated files stored in the specified project directory.
"""
#%%
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
import torch
from TumorNetSolvers.training.updating_trainer import Trainer
from batchgenerators.utilities.file_and_folder_operations import load_json
from set_env import set_environment_variables
set_environment_variables()
nnUNet_preprocessed = os.getenv('nnUNet_preprocessed')

# ============ Configuration ============

# Define the dataset and training configuration
DATASET_NAME = 'Dataset500_Brain'  
TRAINING_CONFIGURATION = '3d_fullres'  # '2d', '3d_lowres', '3d_fullres', etc.
DEVICE = torch.device('cuda:0')  

# Define project and training parameters
PROJECT_NAME = "NN-based-tumor-solvers"  # for wandb
MODEL_NAME = "nnUnet"  # other options are 'TumorSurrogate' and 'ViT'
SIGNATURE = "10k"  # Unique signature for logging and reproducibility

# ============ Load Training Plans and Dataset ============

PLANS_FILE = os.path.join(nnUNet_preprocessed, DATASET_NAME, 'nnUNetPlans.json')
DATASET_JSON_FILE = os.path.join(nnUNet_preprocessed, DATASET_NAME, 'dataset.json')

# Load plans and dataset metadata
print(f"Loading plans from: {PLANS_FILE}")
plan = load_json(PLANS_FILE)

print(f"Loading dataset JSON from: {DATASET_JSON_FILE}")
dataset_json = load_json(DATASET_JSON_FILE)

# ============ Initialize and Run Training ============

# Initialize the nnU-Net trainer
print("Initializing trainer...")
trainer = Trainer(
    plans=plan,
    configuration=TRAINING_CONFIGURATION,
    device=DEVICE,
    signature=SIGNATURE,
    model=MODEL_NAME,
    dataset_json=dataset_json,
    project_name=PROJECT_NAME
)

# Run the training process
print("Starting training...")
trainer.run_training()
print("Training complete.")

# %%
