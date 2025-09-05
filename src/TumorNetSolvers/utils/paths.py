# This script sets up the environment variables for nnU-Net and validates their paths.
"""
import os
import sys

# Define base directory
mount_dir = "/mnt/Drive3/jonas_zeineb/data_and_outputs"

# Set paths for nnU-Net directories
path_dict = {
    "nnUNet_raw": os.path.join(mount_dir, "raw_data"),
    "nnUNet_preprocessed": os.path.join(mount_dir, "preprocessed_data"),
    "nnUNet_results": os.path.join(mount_dir, "results"),
}

# Set environment variables
for env_var, path in path_dict.items():
    os.environ[env_var] = path

# Validate environment variables
for env_var, expected_path in path_dict.items():
    if os.environ.get(env_var) != expected_path:
        print(f"Error: {env_var} is not set correctly.")
        sys.exit(1)

# Check if required paths are set
if not all(os.environ.get(var) for var in path_dict.keys()):
    print("Error: One or more required environment variables are not set.")
    sys.exit(1)

print("All paths and environment variables are set correctly.")
"""
import os

def set_environment_variables():
    mount_dir = "/mnt/Drive3/yeray_jonas/TumorNetSolvers_ext/data_and_outputs"

    # Set environment variables
    os.environ['nnUNet_raw'] = os.path.join(mount_dir, "raw_data")
    os.environ['nnUNet_preprocessed'] = os.path.join(mount_dir, "preprocessed_data")
    os.environ['nnUNet_results'] = os.path.join(mount_dir, "results")

    print("Environment variables set successfully.")

if __name__ == "__main__":
    set_environment_variables()