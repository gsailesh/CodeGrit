
import kagglehub
import os
import sys

# Get API token from environment variable
api_token = os.environ.get("KAGGLE_API_TOKEN")

if not api_token:
    print("Please set the KAGGLE_API_TOKEN environment variable.")
    sys.exit(1)

# Download latest version
path = kagglehub.dataset_download("behrad3d/nasa-cmaps")

print("Path to dataset files:", path)