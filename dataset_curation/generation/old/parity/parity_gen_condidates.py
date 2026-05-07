import os
import cv2
import base64
import math
import asyncio
import json
import re
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

load_dotenv("/home/Pupil/dataset_curation/generation/MMCTAgent/examples/.env")

# Azure & OpenAI Imports
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

# MMCT Imports
from mmct.video_pipeline import VideoAgent 

# ==============================================================================
# 🎛️ KNOBS (CONFIGURATION)
# ==============================================================================

KNOBS = {
    "VIDEO_NAME": "engg_drawing_mod02lec11",
    "TARGET_QUESTION_COUNT": 5,
    
    # Paths
    "BASE_DIR": Path("/home/Pupil/dataset"),
    "VIDEOS_ROOT": Path("/home/Pupil/dataset/videos_db/inital_v2"),
    "TRANSCRIPT_DIR": Path("/home/Pupil/dataset/transcripts_db"),
    "OUTPUT_DIR": Path("/home/Pupil/dataset/queries_db/parity_queries"),
    
    # Models
    "GEN_MODEL": "gpt-5_2025-08-07",        # High reasoning for Q&A Generation
    "UTILITY_MODEL": "gpt-5.1_2025-11-13",     # Reliable for Formatting & Validation logic
    
    "API_VERSION_GEN": "2024-12-01-preview",
    "API_VERSION_UTIL": "2024-12-01-preview",
    
    "FRAMES_TO_EXTRACT": 10,
}

KNOBS["ORIGINAL_VIDEO_PATH"] = KNOBS["VIDEOS_ROOT"] / f"{KNOBS['VIDEO_NAME']}.mp4"
KNOBS["PARITY_VIDEO_PATH"] = KNOBS["VIDEOS_ROOT"] / "parity" / f"{KNOBS['VIDEO_NAME']}.mp4"
KNOBS["OUTPUT_FILE"] = KNOBS["OUTPUT_DIR"] / f"{KNOBS['VIDEO_NAME']}_visual_queries.json"

# ==============================================================================

def get_azure_client(api_version):
    credential = AzureCliCredential()
    token_provider = get_bearer_token_provider(credential, "api://azure/.default")
    return AzureOpenAI(
        azure_endpoint='https://<AZURE_OPENAI_ENDPOINT>',
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )

# Instantiate both clients
gen_client = get_azure_client(KNOBS["API_VERSION_GEN"])
util_client = get_azure_client(KNOBS["API_VERSION_UTIL"])

# --- HELPER FUNCTIONS ---

def format_to_json(raw_agent_output: str) -> list:
    """
    Uses GPT-4o (Utility) to transform messy text into strict JSON.
    """
    print(f"🧹 Formatting output using {KNOBS['UTILITY_MODEL']}...")
    
    prompt = f"""
    You are a data extraction assistant. I have a messy response from a video agent. 
    Extract the questions and answers into a strict JSON list of objects.
    
    Each object must have: "question", "answer", and "source_of_fact": "visual".
    
    RAW OUTPUT:
    {raw_agent_output}
    
    Return ONLY a JSON list. No markdown, no extra text.
    """
    
    # try:
    response = util_client.chat.completions.create(
        model=KNOBS["UTILITY_MODEL"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={ "type": "json_object" } 
    )
    content = response.choices[0].message.content
    print("aa")
    print(content)
    print("aa")
    data = json.loads(content)
    
    # Robust list extraction
    if isinstance(data, dict):
        for val in data.values():
            if isinstance(val, list): return val
        return [data]
    return data
    # except Exception as e:
    #     print(f"    ⚠️ GPT-4o Formatting Error: {e}")
    #     return []

def extract_frames_from_video(video_path: str, num_frames=10) -> list:
    video = cv2.VideoCapture(str(video_path))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: return []
        
    interval = max(1, math.floor(total_frames / num_frames))
    base64_frames = []

    for i in range(0, total_frames, interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = video.read()
        if not success or len(base64_frames) >= num_frames: break
        
        h, w = frame.shape[:2]
        scale = 512 / w
        new_dim = (512, int(h * scale))
        resized = cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(".jpg", resized)
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
    
    video.release()
    return base64_frames

def check_visual_dependency_with_llm(question: str, black_video_path: Path) -> tuple[bool, str]:
    """
    Feeds black frames to GPT-4o (Utility). 
    Uses an LLM judge instead of keyword matching to detect hallucinations.
    """
    print(f"⚖️ Judging visual dependency with {KNOBS['UTILITY_MODEL']}...")
    frames = extract_frames_from_video(black_video_path, KNOBS["FRAMES_TO_EXTRACT"])
    
    # Step 1: Get the model's attempt at the question
    content_payload = [{"type": "text", "text": f"Question: {question}\nAnswer this based on ONLY the video context."}]
    for b64_frame in frames:
        content_payload.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_frame}", "detail": "low"}
        })

    response = util_client.chat.completions.create(
        model=KNOBS["UTILITY_MODEL"],
        messages=[{"role": "user", "content": content_payload}],
        temperature=0.0 
    )
    answer_attempt = response.choices[0].message.content
    print("bb")
    print(answer_attempt)
    print("bb")

    # Step 2: Use the Utility Model to judge if the previous response admitted it couldn't see anything
    judge_prompt = f"""
    Analyze the following response from an AI seeing a video. 
    Did the AI admit the video was black/empty/unviewable, did it say the frames required are missing OR did it try to answer the question anyway?
    
    AI RESPONSE: "{answer_attempt}"
    
    Return JSON: {{"is_visually_grounded": true}} if the AI said it cannot see/video is black/frames missing.
    Return JSON: {{"is_visually_grounded": false}} if the AI provided an actual normal answer.
    """
    
    judge_res = util_client.chat.completions.create(
        model=KNOBS["UTILITY_MODEL"],
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"}
    )
    
    is_valid = json.loads(judge_res.choices[0].message.content).get("is_visually_grounded", False)
    return is_valid, answer_attempt

def load_existing_data():
    if KNOBS["OUTPUT_FILE"].exists():
        with open(KNOBS["OUTPUT_FILE"], "r") as f:
            data = json.load(f)
        path_str = str(KNOBS["ORIGINAL_VIDEO_PATH"])
        return data, data.get(path_str, [])
    return {}, []

def save_data(full_data):
    KNOBS["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
    with open(KNOBS["OUTPUT_FILE"], "w") as f:
        json.dump(full_data, f, indent=2)

# --- MAIN FLOW ---

async def main():
    full_data, final_questions = load_existing_data()
    video_path_str = str(KNOBS["ORIGINAL_VIDEO_PATH"])
    
    if video_path_str not in full_data:
        full_data[video_path_str] = []

    print(f"📂 Video: {KNOBS['VIDEO_NAME']} | Target: {KNOBS['TARGET_QUESTION_COUNT']}")

    while len(final_questions) < KNOBS["TARGET_QUESTION_COUNT"]:
        needed = KNOBS["TARGET_QUESTION_COUNT"] - len(final_questions)
        history_txt = "\n".join([f"- {q['question']}" for q in final_questions]) or "None."

        current_prompt = f"""
        Using GPT-5 reasoning, generate {max(needed, 5)} unique questions that require visual inspection of the video along with their answers.
        The questions must NOT be answerable by transcript/audio alone and should contain a que, slightly hinting at answering using video.
        
        Example: (here visually is the cue)
        Q1: How is soil transferred into the collection bucket during the demonstration\u2014what are the exact steps and actions performed visually?
        A1: The individual in the green coat kneels, manually scoops up loose soil from the sampling pit with their hands, and directly places the soil into the black plastic bucket. This process is performed repeatedly while another person in a checkered shirt assists by holding a large machete-like tool upright, possibly to aid in the digging or keep the area clear.

        and so on...

        Avoid these already accepted questions:
        {history_txt}
        """

        # VideoAgent uses the GEN_MODEL via its internal configuration/index
        video_agent = VideoAgent(
            query=current_prompt, 
            index_name=f"{KNOBS['VIDEO_NAME']}_index",
            use_critic_agent=False,
            stream=False
        )
        
        # try:
        print(f"🤖 Calling MMCT Agent (GPT-5 Reasoning)...")
        agent_response = await asyncio.wait_for(video_agent(), timeout=150.0)
        
        # Extract raw text
        raw_text = str(agent_response)
        if hasattr(agent_response, 'content'):
            raw_text = agent_response.content.response if hasattr(agent_response.content, 'response') else str(agent_response.content)
        
        # --- FORMATTING (GPT-4o) ---
        candidates = format_to_json(raw_text)
        
        if not candidates:
            print("⚠️ Formatting failed. Retrying batch...")
            continue

        # --- VALIDATION (GPT-4o) ---
        for item in candidates:
            if len(final_questions) >= KNOBS["TARGET_QUESTION_COUNT"]: break
            
            q_text = item.get('question')
            if not q_text or any(ex['question'] == q_text for ex in final_questions):
                continue
            
            print(f"🔍 Checking Candidate: {q_text[:70]}...")
            is_grounded, val_ans = check_visual_dependency_with_llm(q_text, KNOBS["PARITY_VIDEO_PATH"])
            
            if is_grounded:
                print(f"   ✅ Grounding Confirmed.")
                item["validator_response"] = val_ans
                final_questions.append(item)
                full_data[video_path_str] = final_questions
                save_data(full_data)
            else:
                print(f"   ❌ Rejected (Hallucination detected on black frames).")

        # except asyncio.TimeoutError:
        #     print("🕒 MMCT Agent timed out. Retrying...")
        # except Exception as e:
        #     print(f"💥 Error: {e}")
        #     await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
    print("\n✨ Pipeline execution complete.")