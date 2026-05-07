import yt_dlp
import os

def download_clean_video(url, save_dir):
    if not os.path.join(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Adding '_clean' to identify files that have been re-processed
    output_template = os.path.join(save_dir, '%(title)s_clean.%(ext)s')

    ydl_opts = {
        # Standardize to 720p for processing efficiency
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        'merge_output_format': 'mp4',
        'outtmpl': output_template,
        
        # This is the "Magic Fix" for Decord/Qwen errors:
        'postprocessor_args': [
                # 1. The "Magic Filter": Forces exactly 30 unique frames per second.
                #    This fills in VFR gaps and drops excess frames from 60fps sources.
                '-filter:v', 'fps=fps=30', 
                
                # 2. Force standard H.264 encoding (most compatible codec for MLLMs)
                '-c:v', 'libx264',
                
                # 3. Force standardized pixel format (prevents "stride" errors in tensors)
                '-pix_fmt', 'yuv420p',
                
                # 4. Clean Audio settings (standard AAC)
                '-c:a', 'aac',
                '-b:a', '128k'
            ],
        
        'retries': 10,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading and Re-encoding: {url}")
            ydl.download([url])
            print("\nDone! File is now 'clean' and should not crash Decord.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    video_url = 'https://www.youtube.com/watch?v=G7wnGeR_69k' 
    download_folder = rf"yt_dwonloader\dwonloads0"
    
    download_clean_video(video_url, download_folder)