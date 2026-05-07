import asyncio
from dotenv import load_dotenv
load_dotenv("/home/Pupil/dataset_curation/generation/MMCTAgent/examples/.env")
from mmct.image_pipeline import ImageAgent, ImageQnaTools
from azure.identity import AzureCliCredential

credential = AzureCliCredential()

# Initialize the Image Agent with desired tools
image_agent = ImageAgent(
    query="which car is that",
    image_path="/home/Pupil/dataset_curation/dataset/car.jpg",
    tools=[ImageQnaTools.object_detection, ImageQnaTools.ocr, ImageQnaTools.vit],
    use_critic_agent=False,  # Enable critical thinking
    stream=False
)

# Run the analysis
response = asyncio.run(image_agent())
print(f"Analysis Result: {response}")
