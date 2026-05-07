import os
import json
import time
import signal
import sys
import yt_dlp

# --- CONFIG ---
WORK_DIR = "/workspace/Pupil/contrastive_experiments/setup"
DATA_ROOT = "/data/Pupil/ActivityNet-QA"
ANNO_DIR = os.path.join(DATA_ROOT, "annotations")
VIDEO_DIR = os.path.join(DATA_ROOT, "videos")
PROGRESS_FILE = os.path.join(WORK_DIR, "download_progress.json")
COOKIES_PATH = "/workspace/Pupil/contrastive_experiments/setup/youtubecom_cookies.txt"

# Target number of valid QA pairs
TARGETS = {
    "train_q.json": 1000,
    "val_q.json": 100,
    "test_q.json": 100
}

QA_PAIRING = {
    "train_q.json": "train_a.json",
    "val_q.json": "val_a.json",
    "test_q.json": "test_a.json"
}

# --- GLOBAL STATE ---
current_data = {
    "q_entries": [],
    "a_entries": [],
    "index": 0,
    "success_count": 0,
    "current_file": ""
}

# --- FFmpeg / yt-dlp Configuration ---
# This applies your requested standardization settings
ydl_opts_base = {
    'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
    'merge_output_format': 'mp4',
    'quiet': True,
    'no_warnings': True,
    'cookiefile': COOKIES_PATH,
    'socket_timeout': 15,
    'retries': 10,
    
    # Passing the ffmpeg arguments to standardizers
    'postprocessor_args': {
        'ffmpeg': [
            '-filter:v', 'fps=fps=30',  # Force 30 fps
            '-c:v', 'libx264',          # H.264 Codec
            '-pix_fmt', 'yuv420p',      # Standard Pixel Format
            '-c:a', 'aac',              # AAC Audio
            '-b:a', '128k'              # Audio Bitrate
        ]
    }
}

def save_progress(final=False):
    """Saves valid entries SO FAR to disk."""
    if not current_data["current_file"]: return

    q_filename = current_data["current_file"]
    dummy_q_name = q_filename.replace('.json', '_dummy.json')
    dummy_a_name = QA_PAIRING[q_filename].replace('.json', '_dummy.json')
    
    with open(os.path.join(ANNO_DIR, dummy_q_name), 'w') as f:
        json.dump(current_data["q_entries"], f, indent=2)
    with open(os.path.join(ANNO_DIR, dummy_a_name), 'w') as f:
        json.dump(current_data["a_entries"], f, indent=2)

    # Save resumption state
    state = {
        "current_file": current_data["current_file"],
        "index": current_data["index"],
        "success_count": current_data["success_count"]
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(state, f)

    if final:
        print(f"\n[Saved] Progress saved. Resumable from index {current_data['index']}.")

def signal_handler(sig, frame):
    print("\n\n!!! KeyboardInterrupt detected !!!")
    save_progress(final=True)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def download_video(video_id):
    """
    Returns:
    0: Success
    1: Permanent Failure (Deleted/Private)
    2: Rate Limit (Wait needed)
    """
    yt_id = video_id[2:] if video_id.startswith('v_') else video_id
    output_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

    # If file exists and is valid size (>1KB), skip
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        return 0

    # Configure output template for this specific video
    current_opts = ydl_opts_base.copy()
    current_opts['outtmpl'] = os.path.join(VIDEO_DIR, f"{video_id}.%(ext)s")

    try:
        with yt_dlp.YoutubeDL(current_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={yt_id}"])
        return 0
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e).lower()
        if "rate-limit" in error_msg or "429" in error_msg:
            return 2
        elif "video unavailable" in error_msg or "private video" in error_msg or "account associated" in error_msg:
            print(f" -> [Failed] {video_id}: Video Unavailable/Private")
            return 1
        else:
            print(f" -> [Error] {video_id}: {e}")
            return 1
    except Exception as e:
        print(f" -> [Critical] {video_id}: {e}")
        return 1

def load_previous_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f: return json.load(f)
    return None

def main():
    if not os.path.exists(ANNO_DIR): os.makedirs(ANNO_DIR)
    if not os.path.exists(VIDEO_DIR): os.makedirs(VIDEO_DIR)
    
    # Load state
    saved_state = load_previous_progress()

    for q_filename, target_count in TARGETS.items():
        start_index = 0
        current_success = 0
        
        # Load existing dummy data
        dummy_q_path = os.path.join(ANNO_DIR, q_filename.replace('.json', '_dummy.json'))
        dummy_a_path = os.path.join(ANNO_DIR, QA_PAIRING[q_filename].replace('.json', '_dummy.json'))
        
        existing_q, existing_a = [], []
        if os.path.exists(dummy_q_path) and os.path.exists(dummy_a_path):
            with open(dummy_q_path, 'r') as f: existing_q = json.load(f)
            with open(dummy_a_path, 'r') as f: existing_a = json.load(f)
            current_success = len(existing_q)

        if current_success >= target_count:
            print(f"Skipping {q_filename} (Target met: {current_success}/{target_count})")
            continue

        if saved_state and saved_state["current_file"] == q_filename:
            start_index = saved_state["index"] + 1 # Resume from NEXT index
            print(f"RESUMING {q_filename} from index {start_index} (Collected: {current_success})")
        else:
            print(f"STARTING {q_filename} (Target: {target_count})")

        current_data.update({
            "q_entries": existing_q, 
            "a_entries": existing_a, 
            "current_file": q_filename, 
            "success_count": current_success
        })

        with open(os.path.join(ANNO_DIR, q_filename), 'r') as f: source_q = json.load(f)
        with open(os.path.join(ANNO_DIR, QA_PAIRING[q_filename]), 'r') as f: source_a = json.load(f)

        for i in range(start_index, len(source_q)):
            if current_data["success_count"] >= target_count:
                print(f"\nTarget reached for {q_filename}!")
                break
            
            current_data["index"] = i
            entry = source_q[i]
            video_name = entry['video_name']
            
            # --- PROGRESS DISPLAY ---
            print(f"[{i}/{len(source_q)}] Scanning {video_name}... (Found: {current_data['success_count']})", end='\r')
            
            status = download_video(video_name)
            
            if status == 0:
                current_data["q_entries"].append(entry)
                current_data["a_entries"].append(source_a[i])
                current_data["success_count"] += 1
                # Clear line and print success
                sys.stdout.write("\033[K") # Clear line
                print(f"[{i}] SUCCESS: {video_name} | Total: {current_data['success_count']}/{target_count}")
                if current_data["success_count"] % 5 == 0: save_progress()
                
            elif status == 2:
                sys.stdout.write("\033[K")
                print(f"[{i}] RATE LIMIT HIT. Saving and sleeping 60s...")
                save_progress()
                time.sleep(60)

        save_progress()
        
    print("\nAll Done.")
    if os.path.exists(PROGRESS_FILE): os.remove(PROGRESS_FILE)

if __name__ == "__main__":
    main()