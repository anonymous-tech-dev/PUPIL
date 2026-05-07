import asyncio
import nest_asyncio
nest_asyncio.apply()
import os
from dotenv import load_dotenv
load_dotenv("/home/Pupil/dataset_curation/generation/MMCTAgent/examples/.env")

from azure.identity import AzureCliCredential
credential = AzureCliCredential()

from mmct.video_pipeline import IngestionPipeline, Languages, TranscriptionServices

video_path = "/home/Pupil/dataset_curation/dataset/videos_db/v1_500/catgA_hydro_stats_mod01lec14.mp4"
keyframe_config = {"motion_threshold": 1.5, "sample_fps": 2} # Example keyframe extraction config

url = "video-url"
source_language = Languages.ENGLISH_UNITED_STATES
transcript_path = f"/home/Pupil/dataset_curation/dataset/transcripts_db/{os.path.splitext(os.path.basename(video_path))[0]}_transcript.srt"
index_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_index"

# Create IngestionPipeline instance
ingestion = IngestionPipeline(
    video_path=video_path,
    index_name=index_name,
    transcription_service=TranscriptionServices.AZURE_STT,
    language=source_language,
    transcript_path=transcript_path, #Optional: provide if external transcript file is available
    keyframe_config=keyframe_config,
    # url=url #Optional: provide if video is from a URL
)

# Run the ingestion pipeline
asyncio.run(ingestion.run())
print("Ingestion pipeline completed successfully!")