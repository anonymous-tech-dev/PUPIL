import yt_dlp
import csv

def export_limited_videos_to_csv(playlists_data, output_file="indexed_videos.csv"):
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'ignoreerrors': True 
    }
    
    # 1. Setup our global trackers
    global_index = 1
    total_seconds_saved = 0 
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # 2. Add 'Index' to the very beginning of our CSV header
        writer.writerow(['Index', 'Playlist Title', 'Video Title', 'Video URL', 'Duration'])
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for item in playlists_data:
                url = item['url']
                max_vids = item['max_videos']
                min_minutes = item.get('min_minutes', 15) 
                min_seconds = min_minutes * 60 
                
                print(f"\nScanning: {url}")
                print(f"Goal: Find up to {max_vids} videos longer than {min_minutes} mins.")
                
                try:
                    info = ydl.extract_info(url, download=False)
                    playlist_title = info.get('title', 'Unknown Playlist')
                    
                    videos_collected = 0 
                    
                    if 'entries' in info:
                        for entry in info['entries']:
                            if videos_collected >= max_vids:
                                print(f"✅ Reached limit of {max_vids} videos for this playlist.")
                                break 
                                
                            if not entry:
                                continue
                            
                            duration = entry.get('duration')
                            
                            if duration and duration > min_seconds:
                                title = entry.get('title', 'Unknown Title')
                                video_url = entry.get('url')
                                
                                mins = int(duration // 60)
                                secs = int(duration % 60)
                                duration_formatted = f"{mins}:{secs:02d}"
                                
                                if video_url:
                                    # 3. Write the global_index into the row
                                    writer.writerow([global_index, playlist_title, title, video_url, duration_formatted])
                                    
                                    # Update all our trackers
                                    videos_collected += 1
                                    total_seconds_saved += duration
                                    
                                    print(f"  [#{global_index}] Added: {title} [{duration_formatted}]")
                                    
                                    # Increase the global index for the next video
                                    global_index += 1
                    
                    if videos_collected < max_vids:
                        print(f"⚠️ Playlist ended. Only found {videos_collected} matching videos.")
                                    
                except Exception as e:
                    print(f"Could not process playlist {url}. Error: {e}")

    # --- Print Summary ---
    total_videos = global_index - 1
    total_hours = int(total_seconds_saved // 3600)
    leftover_mins = int((total_seconds_saved % 3600) // 60)
    
    print(f"\n🎉 All done! Data saved to '{output_file}'")
    print(f"📊 SUMMARY: Extracted {total_videos} videos totaling {total_hours}h {leftover_mins}m of content.")


# --- How to use it ---

# Define your playlists and how many videos you want from each
my_playlists = [
    {
        'url': 'https://www.youtube.com/playlist?list=UUivA7_KLKWo43tFcCkFvydw', 
        'max_videos': 20,
        'min_minutes': 15  
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLWv9VM947MKi_7yJ0_FCfzTBXpQU-Qd3K', 
        'max_videos': 10,
        'min_minutes': 14  
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLTZM4MrZKfW8ukxIuMmf80p3DkpVOHcp4', 
        'max_videos': 10,
        'min_minutes': 14  
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLISEtDmihMo3mpQu4Yp6SLtx-6C57xCOz', 
        'max_videos': 10,
        'min_minutes': 10  
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLb0WW0k29aHrM-KG8JfwKS6qLqUXJCNLg', 
        'max_videos': 10,
        'min_minutes': 14  
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLb0WW0k29aHqFiZ8V4fkeSkTW5QrjiZL6', 
        'max_videos': 10,
        'min_minutes': 15
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PL22w63XsKjqxPO6pQ8wiZcIrtpTznGSre', 
        'max_videos': 15,
        'min_minutes': 15
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLOaNAKtW5HLRrId9DNANr8GzMsoh6QNbE', 
        'max_videos': 10,
        'min_minutes': 15  
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLOaNAKtW5HLRJM6uFHWD5EoXqg5djf0pc', 
        'max_videos': 10,
        'min_minutes': 10
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLg2tfDG3Ww4uVNaAJ-YDwYSmARrQTmWBx', 
        'max_videos': 10,
        'min_minutes': 5  
    },
    {
        'url': 'http://youtube.com/playlist?list=PLg2tfDG3Ww4tpZKRxLnCy0Z04w3JruyN_', 
        'max_videos': 10,
        'min_minutes': 6 
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLRqDfxcafc23LXGoItpkYMKtUdHaQwSDC', 
        'max_videos': 10,
        'min_minutes': 11
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLRqDfxcafc206fNQPkcBUFEMYje-UjtqA', 
        'max_videos': 20,
        'min_minutes': 10 
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLfUGOPyKWwiqnvWAK4yCPJQvbrRf0E2B8', 
        'max_videos': 15,
        'min_minutes': 8 
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLfUGOPyKWwipnRAbDxK-JXEkZNuYlCzYd', 
        'max_videos': 10,
        'min_minutes': 11
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLHeL0JWdJLvTuGCyC3qvx0RM39YvopVQN', 
        'max_videos': 10,
        'min_minutes': 15
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLvgS71fU12Mbx-w18Chu_Sg9v6loipEFO', 
        'max_videos': 26,
        'min_minutes': 15
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLkyBCj4JhHt8CID9Kz9dcxnoK9rg6f9hW', 
        'max_videos': 20,
        'min_minutes': 10
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLkyBCj4JhHt-EKlWZmoaw9V9-qIDdngm6', 
        'max_videos': 20,
        'min_minutes': 10
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLkyBCj4JhHt-80ttR5a_fwtFO4SwDAFld', 
        'max_videos': 20,
        'min_minutes': 10
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLkyBCj4JhHt8DFH9QysGWm4h_DOxT93fb', 
        'max_videos': 20,
        'min_minutes': 10
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLfUGOPyKWwirgG0alVqlRjya7PoDlqrFy', 
        'max_videos': 20,
        'min_minutes': 10
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLbMVogVj5nJTZJHsH6uLCO00I-ffGyBEm', 
        'max_videos': 15,
        'min_minutes': 15
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLbRMhDVUMngePP5JcezxImF-FzOC9wstz', 
        'max_videos': 15,
        'min_minutes': 15
    },
    {
        'url': 'https://www.youtube.com/playlist?list=PLD8E646BAB3366BC8', 
        'max_videos': 15,
        'min_minutes': 15
    }
]

# Run the function
# export_limited_videos_to_csv(my_playlists, output_file="curated_videos.csv")
print("✅ Why are you rnning thid?")