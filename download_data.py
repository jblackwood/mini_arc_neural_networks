import kagglehub
import os
import shutil
import urllib.request
import zipfile
import subprocess
from dotenv import load_dotenv
import datasets

# Load environment variables from .env file
load_dotenv()

# Define the output directory
data_dir = "data"

# Download the-arc-gen-100k-dataset from Kaggle
arcgen_dataset_name = "arcgen100k/the-arc-gen-100k-dataset"
arcgen_dataset_folder = "the-arc-gen-100k-dataset"
arcgen_dataset_path = os.path.join(data_dir, arcgen_dataset_folder)

if os.path.exists(arcgen_dataset_path) and os.listdir(arcgen_dataset_path):
    print(f"Dataset '{arcgen_dataset_name}' already exists in '{arcgen_dataset_path}'. Skipping download.")
else:
    print(f"Downloading dataset '{arcgen_dataset_name}'...")
    
    # Download latest version
    path = kagglehub.dataset_download(arcgen_dataset_name)
    
    print(f"Downloaded to: {path}")
    
    # Create data directory if it doesn't exist
    os.makedirs(arcgen_dataset_path, exist_ok=True)
    
    # Copy contents to data directory
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(arcgen_dataset_path, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    
    print(f"Dataset copied to '{arcgen_dataset_path}' directory.")

# Download re-arc from GitHub
rearc_zip_url = "https://github.com/michaelhodel/re-arc/raw/main/re_arc.zip"
rearc_dataset_folder = "re-arc"
rearc_dataset_path = os.path.join(data_dir, rearc_dataset_folder)

if os.path.exists(rearc_dataset_path) and os.listdir(rearc_dataset_path):
    print(f"re-arc dataset already exists in '{rearc_dataset_path}'. Skipping download.")
else:
    print(f"Downloading re-arc from GitHub...")
    
    # Download the zip file
    rearc_zip_filename = os.path.join(data_dir, "re_arc.zip")
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(rearc_zip_url, rearc_zip_filename)
    
    print(f"Downloaded to: {rearc_zip_filename}")
    
    # Create re-arc directory
    os.makedirs(rearc_dataset_path, exist_ok=True)
    
    # Extract the zip file
    with zipfile.ZipFile(rearc_zip_filename, 'r') as zip_ref:
        zip_ref.extractall(rearc_dataset_path)
    
    # Remove the zip file
    os.remove(rearc_zip_filename)
    
    print(f"re-arc dataset extracted to '{rearc_dataset_path}' directory.")

# Download Kaggle competition data using kaggle CLI
kaggle_competition = "arc-prize-2025"
competition_dataset_path = os.path.join(data_dir, kaggle_competition)

if os.path.exists(competition_dataset_path) and os.listdir(competition_dataset_path):
    print(f"Competition '{kaggle_competition}' already exists in '{competition_dataset_path}'. Skipping download.")
else:
    print(f"Downloading competition '{kaggle_competition}' using Kaggle CLI...")
    
    # Create data directory if it doesn't exist
    os.makedirs(competition_dataset_path, exist_ok=True)
    
    # Download competition data using kaggle CLI
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", kaggle_competition, "-p", competition_dataset_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"Competition data downloaded to '{competition_dataset_path}' directory.")
        
        # Unzip any downloaded files
        for filename in os.listdir(competition_dataset_path):
            if filename.endswith(".zip"):
                zip_path = os.path.join(competition_dataset_path, filename)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(competition_dataset_path)
                os.remove(zip_path)
                print(f"Extracted and removed {filename}")
    else:
        print(f"Error downloading competition: {result.stderr}")

# Download ARC-AGI from GitHub
github_zip_url = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
github_dataset_folder = "ARC-AGI"
github_dataset_path = os.path.join(data_dir, github_dataset_folder)

if os.path.exists(github_dataset_path) and os.listdir(github_dataset_path):
    print(f"ARC-AGI dataset already exists in '{github_dataset_path}'. Skipping download.")
else:
    print(f"Downloading ARC-AGI from GitHub...")
    
    # Download the zip file
    zip_filename = os.path.join(data_dir, "arc-agi-master.zip")
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(github_zip_url, zip_filename)
    
    print(f"Downloaded to: {zip_filename}")
    
    # Extract the zip file
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Rename the extracted folder
    extracted_folder = os.path.join(data_dir, "ARC-AGI-master")
    os.rename(extracted_folder, github_dataset_path)
    
    # Remove the zip file
    os.remove(zip_filename)
    print(f"ARC-AGI dataset extracted to '{github_dataset_path}' directory.")

# Download ARC HEAVY dataset from Hugging Face
heavy_dataset_name = "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems"
heavy_dataset_folder = "arc-heavy"
heavy_dataset_path = os.path.join(data_dir, heavy_dataset_folder)

if os.path.exists(heavy_dataset_path) and os.listdir(heavy_dataset_path):
    print(f"ARC HEAVY dataset already exists in '{heavy_dataset_path}'. Skipping download.")
else:
    print(f"Downloading ARC HEAVY dataset from Hugging Face...")
    
    # Create data directory if it doesn't exist
    os.makedirs(heavy_dataset_path, exist_ok=True)
    
    # Download and save the dataset
    ds = datasets.load_dataset(heavy_dataset_name)
    
    # Save the dataset to the data directory
    ds.save_to_disk(heavy_dataset_path)
    
    print(f"ARC HEAVY dataset saved to '{heavy_dataset_path}' directory.")

# Download ConceptARC from GitHub
conceptarc_zip_url = "https://github.com/victorvikram/ConceptARC/archive/refs/heads/main.zip"
conceptarc_dataset_folder = "ConceptARC"
conceptarc_dataset_path = os.path.join(data_dir, conceptarc_dataset_folder)

if os.path.exists(conceptarc_dataset_path) and os.listdir(conceptarc_dataset_path):
    print(f"ConceptARC dataset already exists in '{conceptarc_dataset_path}'. Skipping download.")
else:
    print(f"Downloading ConceptARC from GitHub...")
    
    # Download the zip file
    conceptarc_zip_filename = os.path.join(data_dir, "conceptarc-main.zip")
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(conceptarc_zip_url, conceptarc_zip_filename)
    
    print(f"Downloaded to: {conceptarc_zip_filename}")
    
    # Extract the zip file
    with zipfile.ZipFile(conceptarc_zip_filename, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Rename the extracted folder
    extracted_folder = os.path.join(data_dir, "ConceptARC-main")
    os.rename(extracted_folder, conceptarc_dataset_path)
    
    # Remove the zip file
    os.remove(conceptarc_zip_filename)
    
    print(f"ConceptARC dataset extracted to '{conceptarc_dataset_path}' directory.")
