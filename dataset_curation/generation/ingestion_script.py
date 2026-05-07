import asyncio
import nest_asyncio
nest_asyncio.apply()

from datetime import datetime

import os
import glob
from dotenv import load_dotenv
load_dotenv("/home/Pupil/dataset_curation/generation/MMCTAgent/examples/.env")

from azure.identity import AzureCliCredential
from mmct.video_pipeline import IngestionPipeline, Languages, TranscriptionServices

# 1. Setup Environment
credential = AzureCliCredential()

# 2. Configuration
VIDEO_DIR = "/home/Pupil/dataset_curation/dataset/videos_db/initial_v3/"
TRANSCRIPT_DIR = "/home/Pupil/dataset_curation/dataset/transcripts_db/"
# You might want to save the logs here or in a dedicated logs folder

KEYFRAME_CONFIG = {"motion_threshold": 1.5, "sample_fps": 2}

# --- CONTROL FLAGS ---
# 1-based indexing (e.g., 1 is the first video). Both are inclusive.
START_INDEX = 1
STOP_INDEX = 1
# ---------------------
LOG_FILE_PATH = f"/home/Pupil/dataset_curation/dataset/batch_summary_from_{START_INDEX}_to_{STOP_INDEX}.txt" 

async def process_video(video_path, current_idx):
    """Encapsulates logic for a single video to keep main loop clean."""
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # transcript_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}_transcript.srt")
    transcript_path = "/home/Pupil/dataset_curation/dataset/transcripts_db/3_perplexing_physics_problems_clean_transcript.srt"
    index_name = f"{base_name}_index"

    print(f"\n--- [{current_idx}] Starting Ingestion: {base_name} ---")

    try:
        # Create IngestionPipeline instance
        ingestion = IngestionPipeline(
            video_path=video_path,
            index_name=index_name,                                                                
            transcription_service=TranscriptionServices.AZURE_STT,
            language=Languages.ENGLISH_UNITED_STATES,
            transcript_path=transcript_path,
            keyframe_config=KEYFRAME_CONFIG,
        )

        # Run the ingestion pipeline
        await ingestion.run()
        print(f"✅ Success: {index_name}")
        return True

    except Exception as e:
        print(f"❌ Failed: {index_name}")
        print(f"Error Details: {e}")
        return False 

async def main():
    # Get all .mp4 files and sort them to ensure Order stays consistent
    all_videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    total_videos = len(all_videos)

    # Validate Indices
    if START_INDEX < 1 or STOP_INDEX > total_videos:
        print(f"Error: Indices out of range. You have {total_videos} videos.")
        return

    # Calculate Python list slicing (0-based)
    slice_start = START_INDEX - 1
    slice_end = STOP_INDEX

    videos_to_process = all_videos[slice_start:slice_end]

    print(f"Processing range {START_INDEX} to {STOP_INDEX} ({len(videos_to_process)} videos)...")

    # --- TRACKING LISTS ---
    successful_videos = []
    failed_videos = []

    for i, video_path in enumerate(videos_to_process):
        global_idx = slice_start + i + 1
        base_name = os.path.basename(video_path)
        
        # Await the process
        success = await process_video(video_path, global_idx)
        
        if success:
            successful_videos.append(base_name)
        else:
            failed_videos.append(base_name)

    # --- GENERATE SUMMARY STRING ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary_lines = []
    summary_lines.append("="*40)
    summary_lines.append(f"PROCESSING SUMMARY - {timestamp}")
    summary_lines.append(f"Range: {START_INDEX} to {STOP_INDEX}")
    summary_lines.append("="*40 + "\n")
    
    summary_lines.append(f"✅ Completed ({len(successful_videos)}):")
    if successful_videos:
        for vid in successful_videos:
            summary_lines.append(f"  - {vid}")
    else:
        summary_lines.append("  (None)")

    summary_lines.append(f"\n❌ Errors ({len(failed_videos)}):")
    if failed_videos:
        for vid in failed_videos:
            summary_lines.append(f"  - {vid}")
    else:
        summary_lines.append("  (None)")
    
    summary_lines.append("\n" + "="*40 + "\n")

    summary_text = "\n".join(summary_lines)

    # 1. Print to console
    print("\n" + summary_text)

    # 2. Write to file (Append mode 'a' keeps history, 'w' overwrites)
    with open(LOG_FILE_PATH, "a") as f:
        f.write(summary_text)

    print(f"📝 Summary saved to: {LOG_FILE_PATH}")

if __name__ == "__main__":
    asyncio.run(main())