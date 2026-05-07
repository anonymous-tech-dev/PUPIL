import os
import json
import shutil
import subprocess
import glob
from typing import Dict, Any

# --------------------------------------------------
#                   Constants
# --------------------------------------------------
# Set to -1 to download the entirety of the video without trimming.
# Set to a positive number (e.g., 30 * 60) to enforce a hard limit in seconds.
HARD_LIMIT_SECONDS = -1

def get_video_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extracts detailed metadata using ffprobe in JSON format.
    Includes: Width, Height, FPS, Codec, Bitrate, Duration.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        file_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Find video stream
        video_stream = next((s for s in data.get("streams", []) if s["codec_type"] == "video"), None)
        format_info = data.get("format", {})
        
        if not video_stream:
            return {"filename": os.path.basename(file_path), "error": "No video stream found"}

        # Calculate FPS safely
        fps_str = video_stream.get("r_frame_rate", "0/0")
        if "/" in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 0
        else:
            fps = float(fps_str)

        return {
            "filename": os.path.basename(file_path),
            "duration_sec": float(format_info.get("duration", 0)),
            "size_bytes": int(format_info.get("size", 0)),
            "codec": video_stream.get("codec_name", "unknown"),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": round(fps, 3),
            "pix_fmt": video_stream.get("pix_fmt", "unknown"),  # Checking for yuv420p
            "bitrate": int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else "variable",
            "was_trimmed": False
        }

    except Exception as e:
        print(f"⚠️ Error probing {file_path}: {e}")
        return {"filename": os.path.basename(file_path), "error": str(e)}

def trim_video_strict(input_path: str, output_path: str, limit_sec: int):
    """
    Trims video using Stream Copy (-c copy).
    CRITICAL: This preserves your original FPS, Bitrate, and Encoding exactly.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-t", str(limit_sec),
        "-c", "copy",       # <--- The magic flag: No re-encoding!
        "-map", "0",        # Copy all streams (audio/video/subs)
        "-loglevel", "error",
        output_path
    ]
    subprocess.run(cmd, check=True)

def process_benchmark_dataset(drive_url: str, output_dir: str, limit_seconds: int):
    """
    Downloads from Drive, enforces length limit (or skips if -1), and generates metadata manifest.
    """
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "temp_raw_downloads")
    os.makedirs(temp_dir, exist_ok=True)
    
    manifest = []
    
    print(f"⬇️  Starting download from Drive: {drive_url}")
    
    # 1. Download Content
    try:
        # gdown automatically handles folder vs file if the URL is correct
        if "/folders/" in drive_url:
            subprocess.run(["gdown", "--folder", drive_url, "-O", temp_dir], check=True)
        else:
            # For single files
            subprocess.run(["gdown", drive_url, "-O", temp_dir], check=True)
    except subprocess.CalledProcessError:
        print("❌ Download failed. Please check your Drive link.")
        return

    # 2. Process Files
    raw_files = glob.glob(os.path.join(temp_dir, "**", "*.mp4"), recursive=True)
    
    if not raw_files:
        print("⚠️ No MP4 files found in the download folder.")
        # Clean up temp folder before exiting
        shutil.rmtree(temp_dir)
        return

    print(f"\n⚙️  Processing {len(raw_files)} videos...")

    for raw_path in raw_files:
        filename = os.path.basename(raw_path)
        final_path = os.path.join(output_dir, filename)
        
        # Step A: Get Initial Metadata
        meta = get_video_metadata(raw_path)
        original_duration = meta.get("duration_sec", 0)
        
        # Step B: Logic for Trimming
        # If limit is -1, skip trimming. Otherwise check if it exceeds the limit + buffer
        if limit_seconds != -1 and original_duration > (limit_seconds + 1):
            print(f"✂️  Trimming {filename} to {limit_seconds}s (Original: {original_duration:.2f}s)")
            
            try:
                trim_video_strict(raw_path, final_path, limit_seconds)
                meta["was_trimmed"] = True
                
                # Re-probe to get the exact final duration (Stream copy cuts at Keyframes)
                final_meta = get_video_metadata(final_path)
                meta["final_duration"] = final_meta.get("duration_sec")
                
            except Exception as e:
                print(f"❌ Error trimming {filename}: {e}")
                continue
        else:
            # Just move the file if it's short enough OR if limit is -1
            status_msg = "Keep full video (-1 flag)" if limit_seconds == -1 else f"Original duration is within limit"
            print(f"✅ Keeping original {filename} ({status_msg}: {original_duration:.2f}s)")
            
            shutil.move(raw_path, final_path)
            meta["final_duration"] = original_duration
            
        manifest.append(meta)

    # 3. Cleanup and Save Manifest
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"\n🎉 Done! Processed {len(manifest)} videos.")
    print(f"📄 Metadata saved to: {manifest_path}")

# --------------------------------------------------
# Execution
# --------------------------------------------------
if __name__ == "__main__":
    # Dependencies check
    # Ensure you have installed: pip install gdown
    
    # UPDATE THESE VARIABLES
    DRIVE_LINK = "https://drive.google.com/drive/folders/1TliX0kyLRHpKkZmXMjsAcQWKpTHHmPON?usp=sharing"
    # DRIVE_LINK = "https://drive.google.com/drive/folders/1zQUQhWt9hRoV3tpnvJhHhkk_7Sis3kCS?usp=sharing"
    OUTPUT_DIRECTORY = "/home/Pupil/dataset_curation/dataset/videos_db/train_vids"
    
    process_benchmark_dataset(DRIVE_LINK, OUTPUT_DIRECTORY, limit_seconds=HARD_LIMIT_SECONDS)