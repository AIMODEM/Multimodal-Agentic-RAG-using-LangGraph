import os
from yt_dlp import YoutubeDL

# Define your local paths
base_dir = "data"
video_dir = os.path.join(base_dir, "raw_videos")
audio_dir = os.path.join(base_dir, "raw_audio")

# Create directories if they don't exist
os.makedirs(video_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)

video_url = "https://www.youtube.com/watch?v=o81Rs-w3YMY"

def download_tesla_media():
    # 1. Download Video (.mp4)
    video_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': f'{video_dir}/tesla_q4_2024.%(ext)s',
    }
    
    # 2. Download Audio (.mp3)
    audio_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{audio_dir}/tesla_q4_2024.%(ext)s',
    }

    with YoutubeDL(video_opts) as ydl:
        print("Downloading Video...")
        ydl.download([video_url])

    with YoutubeDL(audio_opts) as ydl:
        print("Downloading Audio...")
        ydl.download([video_url])

if __name__ == "__main__":
    download_tesla_media()