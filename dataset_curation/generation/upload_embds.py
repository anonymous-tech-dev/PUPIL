import os
import tarfile
from huggingface_hub import HfApi

# --- Configuration ---
SOURCE_DIR = "/home/Pupil/dataset_curation/generation/mmct_embds"              # What you want to pack
OUTPUT_DIR = "/home/Pupil/temp/tarfile"     # <--- NEW: Where to save the tar locally
REPO_ID = "/pupil"                       # Your HF repo
TAR_NAME = "embeddings_backup1.tar"                   # Name of the archive file

# The full absolute path where the tar will be saved
TAR_PATH = os.path.join(OUTPUT_DIR, TAR_NAME)

def create_tar_from_dir(source_dir, tar_path):
    print(f"Creating {tar_path} from {source_dir}...")
    
    # Ensure the output directory actually exists before trying to save a file there
    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    
    with tarfile.open(tar_path, "w") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    print(f"Finished creating {tar_path}")

def upload_to_hf(local_file_path, repo_filename, repo_id):
    api = HfApi()
    print(f"Uploading {local_file_path} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=repo_filename, # This keeps the file at the root of your HF repo
        repo_id=repo_id,
        repo_type="dataset", 
    )
    print(f"Uploaded {repo_filename} successfully.")

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        return

    # 1. Archive the folder and save it to your specified OUTPUT_DIR
    create_tar_from_dir(SOURCE_DIR, TAR_PATH)

    # 2. Upload from that new sandbox path to Hugging Face
    upload_to_hf(TAR_PATH, TAR_NAME, REPO_ID)

if __name__ == "__main__":
    main()