import yt_dlp
import os
import re
import csv

# =============================================================================
# v4: SINGLE yt_dlp CALL PER VIDEO
# =============================================================================
#
# The v3 script hit yt_dlp twice per video:
#   1. extract_info(download=False) — fetch title, duration, fps, width, height
#   2. ydl.download()              — actually download and encode
#
# This is eliminated because:
#   - Title & Duration already exist in the CSV
#   - fps, width, height were fetched only to re-encode to the SAME native
#     values. The -filter:v fps/scale line was a no-op. Dropping it preserves
#     native resolution/fps automatically (ffmpeg copies dimensions unless told
#     otherwise, and libx264 re-encodes at the source dimensions by default).
#   - `bestvideo[height<=NATIVE]+bestaudio` ≡ `bestvideo+bestaudio` since no
#     format exceeds native height.
#   - Skip tracking now uses yt_dlp's built-in `download_archive` — a tiny
#     text file of video IDs. Eliminates the need to pre-fetch the title just
#     to construct an expected file path for an os.path.exists() check.
#
# All VLM encoding parameters are preserved exactly:
#   CRF 18 · preset medium · threads 16 · keyframe every 2s · yuv420p · AAC 128k
# =============================================================================


def parse_duration_to_seconds(duration_str):
    """Converts 'MM:SS' or 'H:MM:SS' from the CSV to total seconds."""
    if not duration_str:
        return 0
    parts = duration_str.strip().split(':')
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return 0
    return 0


def sanitize_filename(title):
    """Cleans the title to be filesystem-safe: lowercase, underscores, alphanumeric only."""
    clean_name = title.lower()
    clean_name = re.sub(r'[\s\-]+', '_', clean_name)
    clean_name = re.sub(r'[^\w]', '', clean_name)
    return clean_name


def download_video(url, save_dir, filename_base, duration):
    """Downloads and encodes a single video with VLM-optimized settings in one yt_dlp call."""
    os.makedirs(save_dir, exist_ok=True)

    output_template = os.path.join(save_dir, f'{filename_base}_clean.mp4')
    archive_file = os.path.join(save_dir, '.download_archive.txt')

    force_keyframes_expr = 'expr:gte(t,n_forced*2)'

    ydl_opts = {
        'format': 'bv*+ba/b',
        'merge_output_format': 'mp4',
        'outtmpl': output_template,

        # Skip tracking: yt_dlp records video IDs here after successful download.
        # Interrupted downloads are NOT marked, so they retry automatically.
        'download_archive': archive_file,

        # 'cookiefile': '/workspace/dwonloader/cookies.txt',

        # --- ANTI-BOT MEASURES (unchanged) ---
        'sleep_interval': 4,
        'max_sleep_interval': 11,
        'sleep_requests': 1.5,

        # --- VLM-OPTIMIZED ENCODING PARAMETERS (unchanged) ---
        # No -filter:v needed — native resolution & fps are preserved by default.
        'postprocessor_args': [
            '-c:v', 'libx264',
            '-crf', '18',
            '-preset', 'medium',
            '-pix_fmt', 'yuv420p',
            '-force_key_frames', force_keyframes_expr,
            '-c:a', 'aac',
            '-b:a', '128k',
            '-threads', '16',
        ],

        'retries': 15,
        'fragment_retries': 15,
        'quiet': False,
    }

    # --- 30-MINUTE CROP (unchanged) ---
    if duration and duration > 1800:
        print(f"  -> Video length ({duration}s) exceeds 30 mins. Capping at first 30 minutes.")
        ydl_opts['download_ranges'] = lambda info, ydl: [{'start_time': 0, 'end_time': 1800}]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(
                f"  -> Downloading: native res @ native fps | "
                f"preset=medium | threads=16 | keyframe every 2s"
            )
            ydl.download([url])
            print("  -> Success.")
    except Exception as e:
        print(f"  -> Error processing {url}: {e}")


if __name__ == "__main__":
    csv_file_path = '/workspace/curated_videos_v3.csv'
    download_folder = "/data/Pupil/Pupil_train"

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for line_number, row in enumerate(reader, 2):
                video_url = row.get("Video URL")
                if not video_url:
                    continue

                raw_title = row.get("Video Title", "Unknown")
                duration_str = row.get("Duration", "")
                duration = parse_duration_to_seconds(duration_str)
                clean_title = sanitize_filename(raw_title)

                print(f"\n--- Video {line_number - 1} | {raw_title} ---")

                download_video(
                    video_url,
                    download_folder,
                    clean_title,
                    duration,
                )

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")