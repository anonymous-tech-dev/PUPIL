import asyncio
import json
import re
import os
import nest_asyncio
from typing import List, Dict, Literal
from pathlib import Path
from dotenv import load_dotenv
load_dotenv("/home/Pupil/dataset_curation/MMCTAgent/examples/.env")

# Azure & OpenAI Imports
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

# MMCT Imports
from mmct.video_pipeline import VideoAgent

# --- CONFIGURATION (Same as before) ---
nest_asyncio.apply()

VIDEO_NAME = "hydro_stats_mod01lec14"
SOURCE_OF_FACT: Literal['audio', 'visual'] = 'audio'
TARGET_QUESTION_COUNT = 5
GPT4O_MODEL_NAME = "gpt-4o"

TRANSCRIPT_DIR = Path("/home/Pupil/dataset/transcripts_db")
OUTPUT_DIR = Path("/home/Pupil/dataset/queries_db/exp_v3/sof_aud/")

# --- CLIENT SETUP (Same as before) ---
credential = AzureCliCredential()
token_provider = get_bearer_token_provider(credential, "api://azure/.default")

gpt_client = AzureOpenAI(
    azure_endpoint="https://<AZURE_OPENAI_ENDPOINT>",
    azure_ad_token_provider=token_provider,
    api_version="2024-10-21"
)

# --- HELPER FUNCTIONS (Same as before) ---
def load_and_clean_srt(video_name: str) -> str:
    srt_path = TRANSCRIPT_DIR / f"{video_name}_transcript.srt"
    if not srt_path.exists():
        raise FileNotFoundError(f"❌ Transcript not found at: {srt_path}")
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    clean_text = re.sub(r'\n\d+\n', '\n', '\n' + content)
    clean_text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', clean_text)
    lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
    return " ".join(lines)

def get_gpt_completion(messages: List[Dict]) -> str:
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-5.1_2025-11-13",
            messages=messages,
            temperature=0.0 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling GPT: {e}")
        return "ERROR"

def llm_judge_correctness(question: str, ground_truth: str, prediction: str) -> bool:
    prompt = f"""
    You are an impartial judge. Compare the Ground Truth Answer with the Model Prediction for the given Question.
    Question: "{question}"
    Ground Truth: "{ground_truth}"
    Model Prediction: "{prediction}"
    Output strictly: "YES" if they match, "NO" if they do not.
    """
    verdict = get_gpt_completion([{"role": "user", "content": prompt}])
    return "YES" in verdict.upper()

def run_knowledge_check(question: str, ground_truth: str) -> bool:
    prompt = f"Answer the following question to the best of your ability. be concise.\n\nQuestion: {question}"
    blind_answer = get_gpt_completion([{"role": "user", "content": prompt}])
    print(f"   ↳ [Blind Ans]: {blind_answer}")
    is_correct = llm_judge_correctness(question, ground_truth, blind_answer)
    if is_correct:
        print("   ❌ Rejected: GPT-5 knew this from pre-training.")
        return False
    else:
        print("   ✅ Passed: GPT-5 could not answer blindly.")
        return True

def run_transcript_check(question: str, ground_truth: str, transcript: str, source_type: str) -> bool:
    prompt = f"""
    Answer the question using ONLY the provided transcript.
    Transcript: "{transcript[:15000]}"
    Question: {question}
    """
    transcript_answer = get_gpt_completion([{"role": "user", "content": prompt}])
    print(f"   ↳ [Transcript Ans]: {transcript_answer}")
    is_found_in_text = llm_judge_correctness(question, ground_truth, transcript_answer)
    
    if source_type == 'audio':
        if is_found_in_text:
            print("   ✅ Passed: Audio fact successfully verified in transcript.")
            return True
        else:
            print("   ❌ Rejected: Source is audio, but answer not found in transcript.")
            return False
    elif source_type == 'visual':
        if is_found_in_text:
            print("   ❌ Rejected: Source is visual, but answer WAS found in transcript.")
            return False
        else:
            print("   ✅ Passed: Visual fact not found in transcript.")
            return True
    return False

def format_to_json_with_gpt4o(raw_text: str, source_type: str) -> List[Dict]:
    """
    Takes raw text from MMCT and uses GPT-4o to structure it into the required JSON format.
    """
    system_prompt = "You are a precise data formatter. Extract questions and answers from the text and output PURE JSON."
    
    user_prompt = f"""
    Refine and format the following raw text into a JSON list of objects.
    
    Raw Text:
    "{raw_text}"
    
    Requirements:
    1. Extract all valid question-answer pairs.
    2. Set "source_of_fact" to "{source_type}" for every item.
    3. Output MUST be a valid JSON list: [ {{ "question": "...", "answer": "...", "source_of_fact": "..." }}, ... ]
    4. Do not include markdown formatting (```json) or extra text. Just the raw JSON string.
    """
    
    try:
        response = gpt_client.chat.completions.create(
            model=GPT4O_MODEL_NAME, # Using the new config variable
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        
        # Clean up markdown if GPT-4o adds it out of habit
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "")
        
        return json.loads(content)
    except Exception as e:
        print(f"⚠️ GPT-4o Formatting Error: {e}")
        return []
    
# --- 5. MAIN PIPELINE (MODIFIED) ---

async def generate_curated_dataset():
    # 1. Load Transcript
    try:
        transcript_text = load_and_clean_srt(VIDEO_NAME)
        print("📄 Transcript loaded successfully.")
    except Exception as e:
        print(e)
        return

    final_questions = []
    index_name = f"{VIDEO_NAME}_index"

    # --- MODIFIED PROMPT TEMPLATE ---
    # Added the {accepted_history} placeholder
    # base_prompt_template = """
    # You are a question generator. Create {num_needed} recall-based questions from the video transcript.
    # Focus strictly on {source_type} details and questions being related to the education domain of the transcript.

    # Already Accepted Questions (DO NOT REPEAT OR REPHRASE THESE):
    # {accepted_history}

    # IMP: Output must be strictly as a JSON list of objects:
    # [
    #   {{ "question": "...", "answer": "...", "source_of_fact": "{source_type}" }}
    # ]
    # """

    base_prompt_template = """
    You are a question generator. Create {num_needed} recall-based questions from the video.
    
    CRITICAL INSTRUCTION:
    Focus strictly on {source_type} details.
    Since the source is 'audio', generate questions based ONLY on what is explicitly spoken.
    
    Already Accepted Questions (to avoid):
    {accepted_history}
    
    Please list the generated questions and their correct answers clearly.
    """

    while len(final_questions) < TARGET_QUESTION_COUNT:
        needed = TARGET_QUESTION_COUNT - len(final_questions)
        print(f"\n--- 🔄 Generating batch... Need {needed} more ---")

        # --- DYNAMICALLY BUILD HISTORY STRING ---
        if final_questions:
            # Create a numbered list of just the question text
            accepted_history = "\n".join([f"- {q['question']}" for q in final_questions])
        else:
            accepted_history = "None so far."

        # Fill the prompt
        current_prompt = base_prompt_template.format(
            num_needed=max(needed + 2, 3), 
            source_type=SOURCE_OF_FACT,
            accepted_history=accepted_history
        )

        # 2. Run MMCT
        video_agent = VideoAgent(
            query=current_prompt,  # Use the dynamic prompt
            index_name=index_name,
            use_critic_agent=False,
            stream=False
        )
        
        try:
            print(f"🤖 Querying MMCT...")
            agent_response = asyncio.run(video_agent())

            if isinstance(agent_response, dict) and 'content' in agent_response:
                raw_content = agent_response['content'].response
            else:
                raw_content = str(agent_response)

            print(" ⏳ Formatting with GPT-4o...")
            candidates = format_to_json_with_gpt4o(raw_content, SOURCE_OF_FACT)
            
            if not candidates:
                print("⚠️ No valid candidates parsed by GPT-4o. Retrying...")
                continue

            # # Clean JSON
            # clean_json_str = raw_content.replace("```json", "").replace("```", "").strip()
            # if "]" in clean_json_str:
            #     clean_json_str = clean_json_str[:clean_json_str.rindex("]") + 1]
            # candidates = json.loads(clean_json_str)
            
        except Exception as e:
            print(f"⚠️ Generation Error: {e}")
            continue

        # 3. Process Candidates
        for item in candidates:
            if len(final_questions) >= TARGET_QUESTION_COUNT:
                break
                
            q_text = item.get('question')
            gt_ans = item.get('answer')
            
            # Sanity check to ensure exact duplicates didn't slip through
            if any(existing['question'] == q_text for existing in final_questions):
                print(f"⚠️ Skipping Duplicate: {q_text}")
                continue

            print(f"\n🔍 Evaluating: {q_text}")
            print(f"   [GT Answer]: {gt_ans}")

            # --- Check 1: Knowledge Check ---
            if not run_knowledge_check(q_text, gt_ans):
                continue

            # --- Check 2: Transcript/Modality Check ---
            if not run_transcript_check(q_text, gt_ans, transcript_text, SOURCE_OF_FACT):
                continue

            # Accepted
            final_questions.append(item)
            print(f"🌟 Question Added! Total: {len(final_questions)}/{TARGET_QUESTION_COUNT}")

    return final_questions

# --- 6. EXECUTION ---

if __name__ == "__main__":
    questions_list = asyncio.run(generate_curated_dataset())
    
    if questions_list:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Define Video Directory for the Key
        VIDEOS_DIR = Path("/home/Pupil/dataset/videos_db")
        video_full_path = str(VIDEOS_DIR / f"{VIDEO_NAME}.mp4")
        
        final_output = {
            video_full_path: questions_list
        }
        
        output_file = OUTPUT_DIR / f"{VIDEO_NAME}_{SOURCE_OF_FACT}_queries.json"
        
        print("\n" + "="*30)
        print(f"💾 Saving to: {output_file}")
        
        with open(output_file, "w") as f:
            json.dump(final_output, f, indent=2)
            
        print("✅ Done!")