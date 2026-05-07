import os
import sys
import subprocess
from pathlib import Path

# ==============================================================================
#                                CONFIG KNOBS
# ==============================================================================

# Default path to use if no command line argument is provided.
# Can be a specific .mp4 file OR a folder containing .mp4 files.
DEFAULT_INPUT_PATH = "/home/Pupil/dataset_curation/dataset/videos_db/v1_500"

# The name of the subdirectory to create for the processed videos.
# It will be created inside the directory where the source video resides.
OUTPUT_SUBDIR_NAME = "parity"

# ------------------------------------------------------------------------------
# FFMPEG Command Template
# ------------------------------------------------------------------------------
# -f lavfi -i color=c=black:s={width}x{height}:r={fps}  -> Generates black video source
# -map 1:v -> Uses the video track from the generated black source (input 1)
# -map 0:a -> Uses the audio track from the original file (input 0)
# -c:a copy -> Copies audio without re-encoding (lossless and fast)
# -tune stillimage -> Optimizes compression for static video (very small file size)
# -shortest -> Stops encoding when the shortest stream (audio) ends
# ------------------------------------------------------------------------------
FFMPEG_CMD = (
    "ffmpeg -y -v error "
    "-i \"{input_path}\" "
    "-f lavfi -i color=c=black:s={width}x{height}:r={fps} "
    "-map 1:v -map 0:a "
    "-c:v libx264 -tune stillimage -pix_fmt yuv420p "
    "-c:a copy "
    "-shortest "
    "\"{output_path}\""
)

# ==============================================================================
#                                MAIN LOGIC
# ==============================================================================

def get_video_metadata(filepath):
    """
    Uses ffprobe to extract width, height, and FPS from the source video
    to ensure the black video matches the original specs.
    """
    try:
        # Get Width
        w = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", 
             "-show_entries", "stream=width", "-of", "csv=s=x:p=0", filepath], text=True
        ).strip()
        
        # Get Height
        h = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", 
             "-show_entries", "stream=height", "-of", "csv=s=x:p=0", filepath], text=True
        ).strip()
        
        # Get FPS (often returns as '30/1' or similar fraction)
        fps_raw = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", 
             "-show_entries", "stream=avg_frame_rate", "-of", "csv=s=x:p=0", filepath], text=True
        ).strip()
        
        # Parse FPS fraction
        if '/' in fps_raw:
            num, den = map(int, fps_raw.split('/'))
            fps = num / den if den != 0 else 30
        else:
            fps = float(fps_raw) if fps_raw else 30

        return w, h, fps

    except Exception as e:
        print(f"  [!] Warning: Could not probe {os.path.basename(filepath)}. Defaulting to 1920x1080 @ 30fps.")
        return "1920", "1080", 30

def process_video(file_path, output_dir):
    """Executes the ffmpeg conversion."""
    file_path = Path(file_path)
    output_file = output_dir / file_path.name
    
    # --- CHECK: Skip if file already exists ---
    if output_file.exists():
        print(f"Skipping: {file_path.name} (Already exists)")
        return
    
    print(f"Processing: {file_path.name}...")

    # 1. Get metadata to match resolution
    width, height, fps = get_video_metadata(str(file_path))

    # 2. Format command
    cmd = FFMPEG_CMD.format(
        input_path=str(file_path),
        output_path=str(output_file),
        width=width,
        height=height,
        fps=fps
    )

    # 3. Run FFMPEG
    try:
        # Running via shell=True so wildcards/paths are handled by shell
        subprocess.run(cmd, shell=True, check=True)
        print(f"  -> Done: {output_file}")
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

    # Create the output directory (e.g., .../parity/)
    parity_dir = output_root / OUTPUT_SUBDIR_NAME
    parity_dir.mkdir(exist_ok=True)

    print(f"--- Starting Processing ---")
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