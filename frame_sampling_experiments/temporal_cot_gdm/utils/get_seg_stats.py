import json
import os
import subprocess
from collections import defaultdict

# --- Configuration ---
JSONL_PATH = "/workspace/Pupil/frame_sampling_experiments/temporal_cot_gdm/video_meta.jsonl"
VIDEO_DIR = "/data/Pupil/lvbench_v2"
NUM_SEGMENTS = 12

def get_video_duration(file_path):
    """Uses ffprobe to get the exact duration of the video in seconds."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of',
            'default=noprint_wrappers=1:nokey=1', file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error reading duration for {file_path}: {e}")
        return None

def parse_time_to_seconds(time_str):
    """Converts HH:MM:SS or MM:SS to total seconds."""
    parts = time_str.strip().split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    else:
        return float(time_str)

def main():
    segment_span_counts = defaultdict(int)
    videos_processed = 0
    total_qas_processed = 0
    missing_videos = 0

    print("Analyzing video intervals. This might take a moment depending on ffprobe speed...\n")

    with open(JSONL_PATH, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            data = json.loads(line)
            video_key = data['key']
            video_path = os.path.join(VIDEO_DIR, f"{video_key}_clean.mp4")

            # Skip if the video isn't downloaded
            if not os.path.exists(video_path):
                missing_videos += 1
                continue

            # Get exact duration
            duration_secs = get_video_duration(video_path)
            
            # Fallback to metadata if ffprobe fails
            if duration_secs is None:
                duration_secs = data['video_info']['duration_minutes'] * 60.0

            segment_length = duration_secs / NUM_SEGMENTS
            videos_processed += 1

            for qa in data.get('qa', []):
                time_ref = qa.get('time_reference')
                if not time_ref:
                    continue

                try:
                    start_str, end_str = time_ref.split('-')
                    start_sec = parse_time_to_seconds(start_str)
                    end_sec = parse_time_to_seconds(end_str)
                    
                    # Prevent out-of-bounds in case timestamps exceed actual video length
                    start_sec = min(start_sec, duration_secs)
                    end_sec = min(end_sec, duration_secs)

                    # Calculate which segment index (0 to 11) the start and end times fall into
                    start_idx = int(start_sec // segment_length)
                    
                    # Subtracting a tiny epsilon from end_sec so exact boundary hits don't spill over
                    end_sec_adj = end_sec - 0.001 if end_sec > start_sec else end_sec
                    end_idx = int(end_sec_adj // segment_length)

                    # Cap at max index 11 (in case of floating point quirks)
                    start_idx = min(start_idx, NUM_SEGMENTS - 1)
                    end_idx = min(end_idx, NUM_SEGMENTS - 1)

                    # Calculate how many segments this interval spans
                    spanned = (end_idx - start_idx) + 1
                    segment_span_counts[spanned] += 1
                    total_qas_processed += 1
                    
                except Exception as e:
                    print(f"Error parsing QA time {time_ref} in video {video_key}: {e}")

    # --- Print Results ---
    print("-" * 40)
    print("📊 SEGMENT SPAN STATISTICS")
    print("-" * 40)
    print(f"Videos Processed: {videos_processed}")
    print(f"Videos Missing:   {missing_videos}")
    print(f"Total QA Pairs:   {total_qas_processed}\n")
    
    print("How many segments do the answers span?")
    for span in sorted(segment_span_counts.keys()):
        count = segment_span_counts[span]
        percentage = (count / total_qas_processed) * 100
        print(f"Spans exactly {span} segment(s): {count:4d} QAs ({percentage:.2f}%)")

if __name__ == "__main__":
    main()