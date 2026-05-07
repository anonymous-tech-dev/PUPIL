import os
import shutil

# CONFIG
SOURCE_DIR = "/home/Pupil/dataset_curation/generation/mmct_embds"
TARGET_DIR = "/home/Pupil/dataset_curation/generation"

# The folders inside each video folder
SUBFOLDERS_TO_MERGE = ["local_storage", "media", "mmct_faiss_indices"]

def merge_directories():
    print(f"🚀 Starting Merge: {SOURCE_DIR} -> {TARGET_DIR}")
    
    # 1. Get all video folders in mmct_embds
    video_folders = [f for f in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, f))]
    total_videos = len(video_folders)
    
    if total_videos == 0:
        print("No video folders found to merge.")
        return

    duplicates_count = 0
    merged_files_count = 0

    for i, vid_folder in enumerate(video_folders):
        print(f"[{i+1}/{total_videos}] Merging data for: {vid_folder}")
        
        vid_path = os.path.join(SOURCE_DIR, vid_folder)
        
        # Loop through the 3 subfolders (local_storage, etc.)
        for sub in SUBFOLDERS_TO_MERGE:
            src_sub = os.path.join(vid_path, sub)
            dst_sub = os.path.join(TARGET_DIR, sub)
            
            # If this video didn't generate one of the folders, skip
            if not os.path.exists(src_sub):
                continue
                
            # Walk through the source directory recursively
            for root, dirs, files in os.walk(src_sub):
                # Construct the corresponding destination path
                relative_path = os.path.relpath(root, src_sub)
                dest_dir = os.path.join(dst_sub, relative_path)
                
                # Ensure destination directory exists
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                
                for file in files:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(dest_dir, file)
                    
                    # --- DUPLICATE CHECK ---
                    if os.path.exists(dst_file):
                        # Optional: Compare file sizes or hashes if you want to be super strict
                        # For now, we assume if it exists, it's a conflict.
                        print(f"  ⚠️ DUPLICATE SKIPPED: {file} already exists in target.")
                        duplicates_count += 1
                    else:
                        shutil.copy2(src_file, dst_file)
                        merged_files_count += 1

    print("\n" + "="*30)
    print("✅ MERGE COMPLETE")
    print(f"Files Merged: {merged_files_count}")
    print(f"Duplicates Skipped: {duplicates_count}")
    print("="*30)

if __name__ == "__main__":
    # Safety confirmation
    confirm = input(f"Merge all folders from {SOURCE_DIR} into {TARGET_DIR}? (y/n): ")
    if confirm.lower() == 'y':
        merge_directories()
    else:
        print("Cancelled.")