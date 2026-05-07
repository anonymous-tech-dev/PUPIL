from dotenv import load_dotenv
load_dotenv("/home/Pupil/dataset_curation/generation/MMCTAgent/examples/.env")
import asyncio
import nest_asyncio
nest_asyncio.apply()

# from azure.identity import AzureCliCredential
# credential = AzureCliCredential()

from mmct.video_pipeline import VideoAgent
# from prompts import QUESTION_GEN_PROMPT_V1

# Test the configuration first
try:
    from mmct.config.settings import MMCTConfig
    config = MMCTConfig()
    print(f"LLM Provider: {config.llm.provider}")
    print(f"LLM Endpoint: {config.llm.endpoint}")
    print(f"LLM Deployment: {config.llm.deployment_name}")
    print(f"Embedding Provider: {config.embedding.provider}")
    print(f"Embedding Endpoint: {config.embedding.endpoint}")
    print(f"Embedding Deployment: {config.embedding.deployment_name}")
    print("✅ Configuration loaded successfully")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    import traceback
    traceback.print_exc()

# measuring-pH-with-Red-Cabbage-480p
# 4min_video
MMCT_PROMPT_TEMPLATE = """
Using the video  describe just 1 question based on the video using your tools and also provide their answers too.

These questions must require looking at the video to answer.

Follow a format like Q1: A1: and so on

Already Accepted Questions (DO NOT REPEAT):
none

Use your tools to get video context
"""
# MMCT_PROMPT_TEMPLATE = """
# Using the video and not the transcript, describe 3 questions based on specific definitions or explanations given in the video using your tools.
# Target concepts where the speaker's definition is unique, specific, or slightly different from a standard textbook definition. (don't mention this though)

# For eg: (not related to this video)
# Q: What is the definition of software engineering?
# A (from video): The definition of software engineering is the practice of ....

# So we will be able to evaluate if a model is seeing the video or answering from its pretrained knowledge.

# Already Accepted Questions (DO NOT REPEAT):
# none

# Use your tools to get context
# """

# MMCT_PROMPT_TEMPLATE = """
# Using GPT-5 reasoning, generate exactly 5 unique questions that require the video to answer, along with their answers.
    
#         Make use of your available tools for more context.

#         Format:
#         Q1: [Question]
#         A1: [Answer]
#         ...

#         Avoid these already generated questions: none yet


# """

# Create VideoAgent instance
video_agent = VideoAgent(
    query=MMCT_PROMPT_TEMPLATE,
    # query="what is this video about?", #"input-query",
    # index_name="catgD_waterlewin_lac5_index", #"your-index-name",
    index_name="3_perplexing_physics_problems_clean_index", #"your-index-name",
    # index_name="catgC_philosophy_mod02lec04_index", #"your-index-name",
    video_id=None,  # Optional: specify video ID
    url=None,  # Optional: URL to filter out the documents
    use_critic_agent=False,  # Enable critic agent
    stream=False,  # Stream response
    cache=False  # Optional: enable caching
)

# Run the agent
response = asyncio.run(video_agent())
print("VideoAgent executed successfully!")

# --- Original Display Code ---
print("\n" + "="*50)
print(response)
print("\n" + "="*50)
print("AGENT SUMMARY RESPONSE")
print("="*50)
print(response['content'].response)

# --- NEW: Displaying Video QnA Details ---
if 'video_qna_response' in response:
    qna_data = response['video_qna_response']
    result = qna_data.get('result', {})

    print("\n" + "="*50)
    print("DETAILED VIDEO QnA DATA")
    print("="*50)

    # 1. Print the structured Q&A text
    if 'answer' in result:
        print("STRATURED Q&A CONTENT:")
        print(result['answer'])

    # 2. Print Token Usage for this specific call
    if 'tokens' in qna_data:
        tokens = qna_data['tokens']
        print(f"\n[Usage] Input Tokens: {tokens.get('total_input')} | Output Tokens: {tokens.get('total_output')}")

    # 3. Print Video Metadata and Timestamps
    if 'videos' in result:
        print("\nVIDEO SOURCE METADATA:")
        for v in result['videos']:
            print(f"- Video Hash: {v.get('hash_id')}")
            if v.get('timestamps'):
                ts_list = [f"[{t[0]} - {t[1]}]" for t in v['timestamps']]
                print(f"  Timestamps: {', '.join(ts_list)}")

# --- Original Sources Used Section ---
print("\n" + "="*50)
print("FINAL SOURCES USED")
print("="*50)
for source in response['content'].source:
    print(f"Video ID: {source.video_id}")
    print("Segments used:")
    for ts in source.timestamps:
        print(f"  - {ts.start_time} to {ts.end_time}")