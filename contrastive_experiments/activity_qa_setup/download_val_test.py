import os
import json
import yt_dlp
import time
import sys

# --- CONFIG ---
DATA_ROOT = "/data/Pupil/ActivityNet-QA"
ANNO_DIR = os.path.join(DATA_ROOT, "annotations")
VIDEO_DIR = os.path.join(DATA_ROOT, "videos")
COOKIES_PATH = "/workspace/Pupil/contrastive_experiments/setup/youtubecom_cookies.txt"

# Target number of QA pairs
TARGETS = {
    "train": 1000,
    "val": 100,
    "test": 100
}

# --- SLOW & SAFE OPTIONS ---
ydl_opts_base = {
    'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
    'merge_output_format': 'mp4',
    'quiet': True,
    'no_warnings': True,
    'cookiefile': COOKIES_PATH,
    'socket_timeout': 30, # Increased timeout
    'retries': 10,
    'postprocessor_args': {
        'ffmpeg': [
            '-filter:v', 'fps=fps=30',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '128k'
        ]
    }
}

def download_video_safe(video_id):
    """
    Returns:
    0: Success (or exists)
    1: Permanent Failure (Deleted/Private)
    2: Bot/Rate Limit (Must Sleep)
    """
    yt_id = video_id[2:] if video_id.startswith('v_') else video_id
    output_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

    # 1. Check if already exists
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        return 0

    print(f"   Attempting download: {video_id}...", end='\r')
    
    current_opts = ydl_opts_base.copy()
    current_opts['outtmpl'] = os.path.join(VIDEO_DIR, f"{video_id}.%(ext)s")

    try:
        with yt_dlp.YoutubeDL(current_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={yt_id}"])
        return 0
    except yt_dlp.utils.DownloadError as e:
        err_msg = str(e).lower()
        if "sign in to confirm" in err_msg or "rate-limit" in err_msg or "429" in err_msg:
            return 2 # Critical Bot Detection
        return 1 # Just a deleted video
    except Exception:
        return 1

def main():
    if not os.path.exists(VIDEO_DIR): os.makedirs(VIDEO_DIR)

    # Process each split
    for split, target_count in TARGETS.items():
        print(f"\n=== SPLIT: {split.upper()} (Target: {target_count}) ===")
        
        q_out = os.path.join(ANNO_DIR, f"{split}_q_1.json")
        a_out = os.path.join(ANNO_DIR, f"{split}_a_1.json")
        
        # 1. SKIP IF DONE
        if os.path.exists(q_out) and os.path.getsize(q_out) > 100:
            with open(q_out, 'r') as f: existing_data = json.load(f)
            if len(existing_data) >= target_count:
                print(f"   [SKIP] {split}_q_1.json already exists with {len(existing_data)} items.")
                continue

        # 2. Load Source
        q_src = os.path.join(ANNO_DIR, f"{split}_q.json")
        a_src = os.path.join(ANNO_DIR, f"{split}_a.json")
        with open(q_src, 'r') as f: q_data = json.load(f)
        with open(a_src, 'r') as f: a_data = json.load(f)

        # 3. Group by Video
        video_groups = {}
        for q, a in zip(q_data, a_data):
            vid = q['video_name']
            if vid not in video_groups: video_groups[vid] = []
            video_groups[vid].append((q, a))

        final_q = []
        final_a = []
        videos_processed = 0

        # 4. Process Videos
        for vid, pairs in video_groups.items():
            if len(final_q) >= target_count:
                break
            
            status = download_video_safe(vid)
            
            if status == 0:
                # Success
                for q_item, a_item in pairs:
                    final_q.append(q_item)
                    final_a.append(a_item)
                videos_processed += 1
                print(f"   [Success] {vid} | Total Pairs: {len(final_q)}/{target_count}     ")
                
                # --- SAFETY SLEEP ---
                # Sleep 20s to look human. 
                # YouTube is less suspicious of "Watch 1 video, wait 20s, Watch next"
                time.sleep(20) 
                
            elif status == 2:
                # Bot Detected
                print(f"\n   [WARNING] Bot detection triggered on {vid}.")
                print("   Sleeping 120 seconds to cool down...")
                time.sleep(120)
                # We skip this video and move to next
            else:
                # Deleted video, just continue immediately
                pass

        # 5. Save
        print(f"\n   Saving {split} to disk...")
        with open(q_out, 'w') as f: json.dump(final_q, f, indent=2)
        with open(a_out, 'w') as f: json.dump(final_a, f, indent=2)

    print("\nAll Done.")

if __name__ == "__main__":
    main()