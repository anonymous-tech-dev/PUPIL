import asyncio
import nest_asyncio
nest_asyncio.apply()

import os
import glob
import shutil
import time
from datetime import datetime
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv("/home/Pupil/dataset_curation/generation/MMCTAgent/examples/.env")
from azure.identity import AzureCliCredential
from mmct.video_pipeline import IngestionPipeline, Languages, TranscriptionServices

# PATHS
PROJECT_ROOT = "/home/Pupil/dataset_curation/generation"
VIDEO_DIR = "/home/Pupil/dataset_curation/dataset/videos_db/train_vids"
TRANSCRIPT_DIR = "/home/Pupil/dataset_curation/dataset/transcripts_db/"
SAFE_STORAGE_DIR = os.path.join(PROJECT_ROOT, "mmct_embds") 

# FOLDERS MMCT EXPECTS
GENERATED_FOLDERS = ["local_storage", "media", "mmct_faiss_indices"]
KEYFRAME_CONFIG = {"motion_threshold": 1.5, "sample_fps": 2}

# Control
START_INDEX = 1
STOP_INDEX = 3

# Log File
LOG_FILE_PATH = f"/home/Pupil/dataset_curation/dataset/batch_summary_from_{START_INDEX}_to_{STOP_INDEX}.txt"

def is_already_processed(video_name):
    target_path = os.path.join(SAFE_STORAGE_DIR, video_name)
    return os.path.exists(target_path)

async def force_cleanup_folders():
    """
    Deletes the generated folders and VERIFIES they are gone.
    """
    print("🧹 Cleaning up stage...")
    for folder in GENERATED_FOLDERS:
        path = os.path.join(PROJECT_ROOT, folder)
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"⚠️ Error deleting {folder}: {e}")
    
    # Wait for OS to catch up
    print("⏳ Waiting 2 minutes for file system to settle after deletion...")
    await asyncio.sleep(120)

    # Verification Loop
    for folder in GENERATED_FOLDERS:
        path = os.path.join(PROJECT_ROOT, folder)
        if os.path.exists(path):
            print(f"⚠️ WARNING: {folder} still exists after cleanup!")
        else:
            # Re-create empty folders immediately so the next run (or current run) doesn't crash
            os.makedirs(path, exist_ok=True)
    
    # Wait for OS to catch up
    await asyncio.sleep(30)
    print("✨ Stage is clean and empty folders are ready.")

async def move_generated_data(video_name):
    """Moves the 3 generated folders to the safe storage."""
    destination_base = os.path.join(SAFE_STORAGE_DIR, video_name)
    if not os.path.exists(destination_base):
        os.makedirs(destination_base)

    for folder in GENERATED_FOLDERS:
        src = os.path.join(PROJECT_ROOT, folder)
        dst = os.path.join(destination_base, folder)
        
        if os.path.exists(src):
            if len(os.listdir(src)) > 0:
                shutil.move(src, dst)
                print(f"📦 Moved {folder} -> {dst}")
            else:
                 print(f"⚠️ {folder} was empty, skipping move.")
        else:
            print(f"⚠️ Warning: {folder} was missing entirely.")

async def process_video(video_path, current_idx):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if is_already_processed(base_name):
        print(f"⏭️  [{current_idx}] Skipping {base_name} (Already in mmct_embds)")
        return "SKIPPED", None # Return Tuple

    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}_transcript.srt")
    index_name = f"{base_name}_index"

    print(f"\n--- [{current_idx}] Processing: {base_name} ---")

    # 1. Prepare Stage
    await force_cleanup_folders()

    try:
        # [Safety Check] Ensure current working dir is correct just in case
        os.chdir(PROJECT_ROOT)

        # 2. Run Pipeline
        ingestion = IngestionPipeline(
            video_path=video_path,
            index_name=index_name,                                                                  
            transcription_service=TranscriptionServices.AZURE_STT, 
            language=Languages.ENGLISH_UNITED_STATES,
            transcript_path=transcript_path,
            keyframe_config=KEYFRAME_CONFIG,
        )
        await ingestion.run()
        
        # 3. Success
        print("✅ Pipeline finished. Sleeping for 2 mins...")
        await asyncio.sleep(120) 

        # 4. Move files
        print("🚚 Moving data...")
        await move_generated_data(base_name)
        return "SUCCESS", None

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Failed: {index_name}")
        print(f"Error Details: {error_msg}")
        
        # 5. Failure Cleanup
        print("⏳ Waiting 10s before cleanup...")
        await asyncio.sleep(10)
        await force_cleanup_folders()
        return "FAILED", error_msg # Return the error string

async def main():
    if not os.path.exists(SAFE_STORAGE_DIR):
        os.makedirs(SAFE_STORAGE_DIR)

    all_videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    
    slice_start = START_INDEX - 1
    slice_end = STOP_INDEX
    videos_to_process = all_videos[slice_start:slice_end]

    print(f"Targeting {len(videos_to_process)} videos...")
    batch_start_time = datetime.now()

    # Tracking
    results = {"SUCCESS": [], "FAILED": [], "SKIPPED": []}
    error_details = {} # Dictionary to store error messages

    for i, video_path in enumerate(videos_to_process):
        base_name = os.path.basename(video_path)
        
        # Capture the status AND the error message
        status, error_msg = await process_video(video_path, slice_start + i + 1)
        
        results[status].append(base_name)
        if status == "FAILED":
            error_details[base_name] = error_msg

    batch_end_time = datetime.now()

    # --- LOGGING SUMMARY ---
    # Format the times for cleaner reading
    start_str = batch_start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_str = batch_end_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate duration (Optional but helpful)
    duration = batch_end_time - batch_start_time
        
    # --- LOGGING SUMMARY ---
    summary_lines = [
        "="*40,
        f"PROCESSING SUMMARY",
        f"Range: {START_INDEX} to {STOP_INDEX}",
        f"Started:  {start_str}",  # [ADDED]
        f"Finished: {end_str}",    # [ADDED]
        f"Duration: {duration}",    # [ADDED]
        "="*40,
        f"\n✅ Completed ({len(results['SUCCESS'])}):"
    ]
    summary_lines.extend([f"  - {v}" for v in results['SUCCESS']] if results['SUCCESS'] else ["  (None)"])
    
    summary_lines.append(f"\n❌ Errors ({len(results['FAILED'])}):")
    if results['FAILED']:
        for v in results['FAILED']:
            # Log the video name
            summary_lines.append(f"  - {v}")
            # Log the exact error message underneath
            summary_lines.append(f"    └── ERR: {error_details.get(v, 'Unknown Error')}")
    else:
        summary_lines.append("  (None)")

    summary_lines.append(f"\n⏭️  Skipped ({len(results['SKIPPED'])}):")
    summary_lines.extend([f"  - {v}" for v in results['SKIPPED']] if results['SKIPPED'] else ["  (None)"])
    
    summary_lines.append("\n" + "="*40 + "\n")
    summary_text = "\n".join(summary_lines)

    print("\n" + summary_text)

    with open(LOG_FILE_PATH, "a") as f:
        f.write(summary_text)

    print(f"📝 Summary saved to: {LOG_FILE_PATH}")

if __name__ == "__main__":
    asyncio.run(main())