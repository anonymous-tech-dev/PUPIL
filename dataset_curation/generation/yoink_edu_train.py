import os
import tarfile
import shutil

def extract_video_chunk(tar_path, output_dir, chunk_index, chunk_size=20):
    """
    Surgically extracts a specific chunk of videos from a large tar file.
    If chunk_index is -1, it extracts ALL videos.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔍 Opening {tar_path}...")
    
    with tarfile.open(tar_path, "r") as tar:
        # 1. Get all mp4 files and sort them alphabetically by their actual file name
        print("Scanning and sorting .mp4 files...")
        members = [m for m in tar.getmembers() if m.name.endswith('.mp4')]
        members.sort(key=lambda m: os.path.basename(m.name))
        
        total_videos = len(members)
        print(f"Found {total_videos} total videos.")

        # 2. Determine which videos to extract
        if chunk_index == -1:
            print(f"\n📦 Extracting ALL {total_videos} videos...")
            chunk = members
        else:
            # Calculate our slice based on the index
            start_idx = chunk_index * chunk_size
            end_idx = start_idx + chunk_size
            
            # Prevent index out of bounds if we hit the end of the list
            chunk = members[start_idx:end_idx]
            
            if not chunk:
                print(f"⚠️ Index {chunk_index} is out of bounds (no videos left).")
                return

            print(f"\n📦 Extracting chunk index {chunk_index} (Videos {start_idx + 1} to {min(end_idx, total_videos)})...")

        # 3. Extract only the files in our chunk
        for member in chunk:
            # We only want the file name, not the whole /content/drive/MyDrive/... path
            clean_filename = os.path.basename(member.name)
            output_path = os.path.join(output_dir, clean_filename)
            
            print(f"Extracting: {clean_filename}")
            
            # Read from the tar and write to the local folder safely
            source_file = tar.extractfile(member)
            if source_file:
                with open(output_path, "wb") as target_file:
                    shutil.copyfileobj(source_file, target_file)

    print(f"\n✅ Extraction complete! Videos saved to: {output_dir}")

# --- HOW TO USE IT ---

# The large tar file you have on your VM
TAR_FILE = "/home/Pupil/temp/edubench_tars/edu_vids_tar_sandbox.tar"

# Where you want the videos to go
OUTPUT_FOLDER = "/home/Pupil/dataset_curation/dataset/videos_db/train_vids"

# Change this index to get different batches! 
# -1 = ALL videos | 0 = vids 1-20 | 1 = vids 21-40 | 2 = vids 41-60
INDEX_TO_EXTRACT = -1

extract_video_chunk(
    tar_path=TAR_FILE, 
    output_dir=OUTPUT_FOLDER, 
    chunk_index=INDEX_TO_EXTRACT
)