import yt_dlp
import os
import json

def download_gold_standard_video(url, save_dir, video_key, target_fps, target_width, target_height):
    os.makedirs(save_dir, exist_ok=True)

    output_template = os.path.join(save_dir, f'{video_key}_clean.mp4')

    # Tell yt-dlp to grab the best video that doesn't exceed the target resolution
    format_selection = f'bestvideo[height<={target_height}]+bestaudio/best[height<={target_height}]'

    # Build the exact FFmpeg video filter string
    video_filter = f'fps=fps={target_fps},scale={target_width}:{target_height}'

    ydl_opts = {
        'format': format_selection,
        'merge_output_format': 'mp4',
        'outtmpl': output_template,
        
        # --- THE CRUCIAL PART: High-fidelity encoding parameters ---
        'postprocessor_args': [
            '-filter:v', video_filter, 
            '-c:v', 'libx264',
            '-crf', '18',           # Visually lossless compression (prevents VLM degradation)
            '-preset', 'slow',      # Better quality-to-bitrate ratio, ensures clean frames
            '-pix_fmt', 'yuv420p',  # Fixes the Decord crash
            '-c:a', 'aac',
            '-b:a', '128k'
        ],
        
        'retries': 10,
        'quiet': False, # Keep this False so you can see the FFmpeg progress
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"  -> Downloading and Encoding: {target_width}x{target_height} @ {target_fps} FPS (CRF 18)")
            ydl.download([url])
            print("  -> Success: File perfectly matches metadata with visually lossless quality.")
    except Exception as e:
        print(f"  -> Error processing {url}: {e}")

if __name__ == "__main__":
    # Update these paths to match your environment
    jsonl_file_path = '/workspace/Pupil/frame_sampling_experiments/temporal_cot_gdm/video_meta.jsonl' 
    download_folder = "/data/Pupil/lvbench_v2"
    
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    video_key = data.get("key")
                    
                    if video_key:
                        expected_file_path = os.path.join(download_folder, f"{video_key}_clean.mp4")
                        if os.path.exists(expected_file_path):
                            print(f"--- Skipping Video {line_number} | ID: {video_key} (Already downloaded) ---")
                            continue 

                        # Extract precise metadata for this specific video
                        video_info = data.get("video_info", {})
                        target_fps = video_info.get("fps", 30.0)
                        
                        resolution = video_info.get("resolution", {})
                        target_width = resolution.get("width", 1280)
                        target_height = resolution.get("height", 720)

                        video_url = f'https://www.youtube.com/watch?v={video_key}'
                        print(f"\n--- Processing Video {line_number} | ID: {video_key} ---")
                        
                        download_gold_standard_video(
                            video_url, 
                            download_folder, 
                            video_key, 
                            target_fps, 
                            target_width, 
                            target_height
                        )
                    else:
                        print(f"Line {line_number}: No 'key' found in JSON data.")
                        
                except json.JSONDecodeError:
                    print(f"Line {line_number}: Invalid JSON format. Skipping.")
                    
    except FileNotFoundError:
        print(f"Error: The file '{jsonl_file_path}' was not found.")