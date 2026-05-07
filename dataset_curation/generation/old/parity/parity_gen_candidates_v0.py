import os
import asyncio
import json
from pathlib import Path
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
    "VIDEO_NAME": "catgA_hydro_stats_mod01lec37",
    "TARGET_TOTAL_COUNT": 20,  # Total desired questions
    "BATCH_SIZE": 5,           # How many to generate per request
    
    # Paths
    "VIDEOS_ROOT": Path("/home/Pupil/dataset_curation/dataset/videos_db/v1_500/"),
    "OUTPUT_DIR": Path("/home/Pupil/dataset_curation/dataset/queries_db/v1_500/parity"),
    
    # Models
    "GEN_MODEL": "gpt-5_2025-08-07",            # High reasoning for Generation
    "UTILITY_MODEL": "gpt-5.1_2025-11-13",      # Reliable for Formatting
    
    "API_VERSION_GEN": "2024-12-01-preview",
    "API_VERSION_UTIL": "2024-12-01-preview",
}

KNOBS["ORIGINAL_VIDEO_PATH"] = KNOBS["VIDEOS_ROOT"] / f"{KNOBS['VIDEO_NAME']}.mp4"
KNOBS["OUTPUT_FILE"] = KNOBS["OUTPUT_DIR"] / f"{KNOBS['VIDEO_NAME']}_visual_parity_queries.json"

# ==============================================================================

def get_azure_client(api_version):
    credential = AzureCliCredential()
    token_provider = get_bearer_token_provider(credential, "api://azure/.default")
    return AzureOpenAI(
        azure_endpoint='https://<AZURE_OPENAI_ENDPOINT>',
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )

# Instantiate Utility Client (Gen client is handled inside VideoAgent)
util_client = get_azure_client(KNOBS["API_VERSION_UTIL"])

# --- HELPER FUNCTIONS ---

def format_to_json(raw_agent_output: str) -> list:
    """
    Uses Utility Model to transform messy text into strict JSON.
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
    
    try:
        response = util_client.chat.completions.create(
            model=KNOBS["UTILITY_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={ "type": "json_object" } 
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        
        # Robust list extraction
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list): return val
            return [data]
        return data
    except Exception as e:
        print(f"⚠️ Formatting Error: {e}")
        return []

def save_data(full_data):
    KNOBS["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
    with open(KNOBS["OUTPUT_FILE"], "w") as f:
        json.dump(full_data, f, indent=2)

def load_existing_data():
    if KNOBS["OUTPUT_FILE"].exists():
        with open(KNOBS["OUTPUT_FILE"], "r") as f:
            data = json.load(f)
        path_str = str(KNOBS["ORIGINAL_VIDEO_PATH"])
        return data, data.get(path_str, [])
    return {}, []

# --- MAIN FLOW ---

async def main():
    full_data, current_questions = load_existing_data()
    video_path_str = str(KNOBS["ORIGINAL_VIDEO_PATH"])
    
    if video_path_str not in full_data:
        full_data[video_path_str] = []

    print(f"📂 Video: {KNOBS['VIDEO_NAME']} | Target: {KNOBS['TARGET_TOTAL_COUNT']}")

    # Clean existing data to prevent crashes from previous bad runs
    current_questions = [q for q in current_questions if "question" in q and "answer" in q]

    while len(current_questions) < KNOBS["TARGET_TOTAL_COUNT"]:
        
        remaining = KNOBS["TARGET_TOTAL_COUNT"] - len(current_questions)
        batch_target = min(remaining, KNOBS["BATCH_SIZE"])
        
        print(f"\n⚙️  Batch Start: Generating {batch_target} questions (Current Total: {len(current_questions)})")

        # SAFE history generation (uses .get just in case, though we cleaned the list above)
        history_txt = "\n".join([f"- {q.get('question', 'UNKNOWN')}" for q in current_questions]) or "None."

        current_prompt = f"""
        Using GPT-5 reasoning, generate exactly {batch_target} unique questions that require the video to answer, along with their answers.
    
        Make use of your available tools for more context.

        Format:
        Q1: [Question]
        A1: [Answer]
        ...

        Avoid these already generated questions:
        {history_txt}
        """

        video_agent = VideoAgent(
            query=current_prompt, 
            index_name=f"{KNOBS['VIDEO_NAME']}_index",
            use_critic_agent=False,
            stream=False
        )
        
        try:
            print(f"🤖 Calling MMCT Agent...")
            agent_response = await asyncio.wait_for(video_agent(), timeout=200.0)

            print("&&&&&&&&&&")
            print(agent_response)
            print("&&&&&&&&&&")
            
            # Extract raw text
            raw_text = str(agent_response)
            if hasattr(agent_response, 'content'):
                raw_text = agent_response.content.response if hasattr(agent_response.content, 'response') else str(agent_response.content)
            
            # --- FORMATTING ---
            new_batch = format_to_json(raw_text)
            
            if not new_batch:
                print("⚠️  No valid JSON returned from formatter. Retrying...")
                continue

            # --- VALIDATION LOOP ---
            added_count = 0
            for item in new_batch:
                if len(current_questions) >= KNOBS["TARGET_TOTAL_COUNT"]: break
                
                # 1. Check if 'question' key actually exists
                if "question" not in item:
                    # Fallback: check case-insensitive keys or skip
                    if "Question" in item: item["question"] = item.pop("Question")
                    else:
                        print(f"⚠️  Skipping invalid item (missing 'question' key): {item}")
                        continue

                # 2. Check if 'answer' key exists
                if "answer" not in item:
                     if "Answer" in item: item["answer"] = item.pop("Answer")
                     else: item["answer"] = "No answer provided."

                # 3. Duplicate check
                if not any(ex['question'] == item['question'] for ex in current_questions):
                    current_questions.append(item)
                    added_count += 1
            
            full_data[video_path_str] = current_questions
            save_data(full_data)
            
            print(f"✅ Batch Complete. Added {added_count} new questions.")

        except asyncio.TimeoutError:
            print("🕒 MMCT Agent timed out. Retrying...")
        except Exception as e:
            print(f"💥 Error in batch: {e}")
            await asyncio.sleep(2)

    print(f"\n✨ Generation Complete! {len(current_questions)} questions saved to {KNOBS['OUTPUT_FILE']}")

if __name__ == "__main__":
    asyncio.run(main())