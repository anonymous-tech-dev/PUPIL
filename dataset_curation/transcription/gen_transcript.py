#!/usr/bin/env python3

# =========================
#         IMPORTS
# =========================
import os
import subprocess
import time
from faster_whisper import WhisperModel
from tqdm import tqdm

# =========================
#          KNOBS
# =========================
# INPUT: Point to a specific MP4 file OR a specific Folder containing MP4s
INPUT_PATH = "/home/Pupil/dataset_curation/dataset/videos_db/final_1k" 

# MODEL SETTINGS
WHISPER_MODEL = "large-v3"      # large-v3 for best accuracy
DEVICE = "cuda"                 # "cuda" for A100
# COMPUTE_TYPE = "float16"        # float16 is standard for A100
COMPUTE_TYPE = "bfloat16"        # float16 is standard for A100

# =========================
#      HELPER FUNCTIONS
# =========================

def get_output_paths(input_video_path):
    """
    Generates dynamic paths for audio and transcript based on the video location.
    Assumes structure: 
       .../dataset/videos_db/vid.mp4
       .../dataset/audio_db/vid.wav
       .../dataset/transcripts_db/vid_transcript.srt
    """
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(input_video_path)))
    
    # 1. Define Audio Path
    audio_dir = os.path.join(parent_dir, "audio_db")
    os.makedirs(audio_dir, exist_ok=True) # Create if missing
    output_wav = os.path.join(audio_dir, f"{base_name}_audio.wav")
    
    # 2. Define Transcript Path
    transcript_dir = os.path.join(parent_dir, "transcripts_db")
    os.makedirs(transcript_dir, exist_ok=True) # Create if missing
    output_srt = os.path.join(transcript_dir, f"{base_name}_transcript.srt")
    
    return output_wav, output_srt

def convert_to_wav(input_path, output_wav):
    """
    Converts valid video/audio to 16kHz mono WAV for Whisper.
    """
    # Check if clean WAV already exists (Cache)
    if os.path.exists(output_wav):
        if os.path.getmtime(output_wav) > os.path.getmtime(input_path):
            print(f"    ✅ Found cached clean audio: {output_wav}")
            return
        else:
            print(f"    ⚠️ Cached WAV is older than source. Re-converting...")

    print(f"    🔄 Converting to clean WAV...")
    
    cmd = [
        "ffmpeg", "-y",                # Overwrite if exists
        "-i", input_path,              # Input
        "-ar", "16000",                # 16kHz sample rate
        "-ac", "1",                    # Mono channel
        "-c:a", "pcm_s16le",           # Codec
        "-vn",                         # No video
        output_wav
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("\n❌ FFmpeg conversion failed!")
        print("Error log:")
        print(e.stderr.decode())
        raise RuntimeError(f"Could not convert {input_path}")

def format_timestamp(seconds):
    """Converts seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def write_srt(segments, output_path, total_duration):
    """Writes segments to SRT while updating a progress bar."""
    with tqdm(total=total_duration, unit="s", desc="    Transcribing", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]") as pbar:
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, seg in enumerate(segments, start=1):
                f.write(f"{idx}\n")
                f.write(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n")
                f.write(f"{seg.text.strip()}\n\n")
                
                current_pos = seg.end
                if current_pos > pbar.n:
                    pbar.update(current_pos - pbar.n)
        
        if pbar.n < total_duration:
            pbar.update(total_duration - pbar.n)

# =========================
#        MAIN LOGIC
# =========================

def main():
    start_global = time.time()
    
    # 1. Determine Files to Process
    files_to_process = []
    
    if os.path.isfile(INPUT_PATH):
        files_to_process.append(INPUT_PATH)
    elif os.path.isdir(INPUT_PATH):
        # Gather all MP4s in the folder and sort them
        files_to_process = sorted([
            os.path.join(INPUT_PATH, f) for f in os.listdir(INPUT_PATH) if f.lower().endswith(".mp4")
        ])
    else:
        print(f"❌ Error: Input path not found: {INPUT_PATH}")
        return

    total_files = len(files_to_process)
    print(f"📂 Found {total_files} file(s) to process.")

    if total_files == 0:
        return

    # 2. Load Model (ONCE for all files)
    print(f"🚀 Loading {WHISPER_MODEL} on {DEVICE}...")
    model = WhisperModel(
        WHISPER_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )

    # 3. Processing Loop
    for i, video_path in enumerate(files_to_process):
        file_start_time = time.time()
        file_num = i + 1
        filename = os.path.basename(video_path)
        
        print(f"\n[{file_num}/{total_files}] Processing: {filename}")
        
        # Calculate dynamic paths based on current file
        audio_path, srt_path = get_output_paths(video_path)

        # ==========================================
        #  [NEW CODE] Check if Transcript Exists
        # ==========================================
        if os.path.exists(srt_path):
            print(f"    ✅ Transcript already exists. Skipping: {os.path.basename(srt_path)}")
            continue
        # ==========================================

        try:
            # Step A: Convert to Wav
            convert_to_wav(video_path, audio_path)

            # Step B: Initialize Transcription
            segments, info = model.transcribe(audio_path, beam_size=5)
            
            print(f"    ℹ️  Detected language: {info.language} ({info.language_probability:.0%})")
            
            # Step C: Write SRT
            write_srt(segments, srt_path, info.duration)
            
            print(f"    ✅ Done. Saved to: {os.path.basename(srt_path)}")
            
        except Exception as e:
            print(f"    ❌ Failed to process {filename}: {e}")
            continue

    # 4. Final Summary
    total_time = time.time() - start_global
    print(f"\n=======================================")
    print(f"🎉 All {total_files} files processed!")
    print(f"⚡ Total workflow time: {total_time:.2f}s")
    print(f"=======================================")

if __name__ == "__main__":
    main()