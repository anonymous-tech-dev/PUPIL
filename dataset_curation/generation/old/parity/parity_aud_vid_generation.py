import os
import sys
import subprocess
from pathlib import Path

# ==============================================================================
#                                CONFIG KNOBS
# ==============================================================================

# Default path to use if no command line argument is provided.
DEFAULT_INPUT_PATH = "/home/Pupil/dataset_curation/dataset/videos_db/v1_500"

# The name of the subdirectory to create for the processed videos.
OUTPUT_SUBDIR_NAME = "parity_silent"

# ------------------------------------------------------------------------------
# FFMPEG Command Template
# ------------------------------------------------------------------------------
# -f lavfi -i color=... -> Generates black video source
# -t {duration} -> Sets the duration to match the original video exactly
# -c:v libx264 ... -> Encodes the video
# -an -> Removes audio entirely (Audio None)
# ------------------------------------------------------------------------------
FFMPEG_CMD = (
    "ffmpeg -y -v error "
    "-f lavfi -i color=c=black:s={width}x{height}:r={fps} "
    "-t {duration} "
    "-c:v libx264 -tune stillimage -pix_fmt yuv420p "
    "-an "
    "\"{output_path}\""
)

# ==============================================================================
#                                MAIN LOGIC
# ==============================================================================

def get_video_metadata(filepath):
    """
    Uses ffprobe to extract width, height, FPS, and DURATION from the source.
    """
    try:
        # Helper to run ffprobe command
        def probe(entry, stream_select=True):
            cmd = ["ffprobe", "-v", "error"]
            if stream_select:
                cmd.extend(["-select_streams", "v:0"])
            cmd.extend(["-show_entries", entry, "-of", "csv=s=x:p=0", filepath])
            return subprocess.check_output(cmd, text=True).strip()

        # Get Width
        w = probe("stream=width")
        
        # Get Height
        h = probe("stream=height")
        
        # Get FPS
        fps_raw = probe("stream=avg_frame_rate")
        if '/' in fps_raw:
            num, den = map(int, fps_raw.split('/'))
            fps = num / den if den != 0 else 30
        else:
            fps = float(fps_raw) if fps_raw else 30

        # Get Duration (Use container format duration for best accuracy)
        dur = probe("format=duration", stream_select=False)
        
        return w, h, fps, dur

    except Exception as e:
        print(f"  [!] Warning: Could not probe {os.path.basename(filepath)}. Defaulting to 1920x1080 @ 30fps (10s).")
        return "1920", "1080", 30, "10"

def process_video(file_path, output_dir):
    """Executes the ffmpeg conversion."""
    file_path = Path(file_path)
    output_file = output_dir / file_path.name
    
    # --- CHECK: Skip if file already exists ---
    if output_file.exists():
        print(f"Skipping: {file_path.name} (Already exists)")
        return
    
    print(f"Processing: {file_path.name}...")

    # 1. Get metadata (now includes duration)
    width, height, fps, duration = get_video_metadata(str(file_path))

    # 2. Format command
    # Note: We do not need input_path in ffmpeg anymore, just the metadata
    cmd = FFMPEG_CMD.format(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        output_path=str(output_file)
    )

    # 3. Run FFMPEG
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"  -> Done: {output_file} ({duration}s)")
    except subprocess.CalledProcessError as e:
        print(f"  [X] Error processing {file_path.name}: {e}")

def main():
    # Priority: Command line arg > Knob at top of script
    target_input = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_PATH
    target_path = Path(target_input)

    if not target_path.exists():
        print(f"Error: Path not found: {target_path}")
        return

    files_to_process = []
    output_root = None

    # Logic to handle File vs Folder
    if target_path.is_file():
        if target_path.suffix.lower() == ".mp4":
            files_to_process.append(target_path)
            output_root = target_path.parent
        else:
            print("Error: The provided file is not an .mp4")
            return
    elif target_path.is_dir():
        files_to_process = list(target_path.glob("*.mp4"))
        output_root = target_path
    
    if not files_to_process:
        print("No .mp4 files found to process.")
        return

    # Create the output directory
    parity_dir = output_root / OUTPUT_SUBDIR_NAME
    parity_dir.mkdir(exist_ok=True)

    print(f"--- Starting Processing (Black Video + No Audio) ---")
    print(f"Target: {target_input}")
    print(f"Output: {parity_dir}")
    print(f"Files found: {len(files_to_process)}")
    print("-" * 30)

    for vid in files_to_process:
        process_video(vid, parity_dir)

    print("-" * 30)
    print("All tasks completed.")

if __name__ == "__main__":
    main()