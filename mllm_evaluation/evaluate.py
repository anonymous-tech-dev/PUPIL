import os
import json
import glob
from tqdm import tqdm
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

# ==============================================================================
#                                   KNOBS
# ==============================================================================

# 1. POINT DIRECTLY TO THE FOLDER CONTAINING JSON FILES
# Use the absolute path to avoid ambiguity
TARGET_DIR = "/home/Pupil/mllm_evaluation/results/videollama3/parity/normal_prompt/parity_vid_silent"

# 2. Re-evaluate logic
OVERWRITE_EXISTING_JUDGEMENTS = True  # If False, skips if "judge_verdict" is already present

# ==============================================================================
#                                AZURE SETUP
# ==============================================================================

try:
    credential = AzureCliCredential()
    token_provider = get_bearer_token_provider(credential, "api://azure/.default")

    gpt_client = AzureOpenAI(
        azure_endpoint="https://<AZURE_OPENAI_ENDPOINT>",
        azure_ad_token_provider=token_provider,
        api_version="2024-10-21"
    )
    print("✅ Azure Client Initialized")
except Exception as e:
    print(f"⚠️ Warning: Failed to initialize Azure OpenAI Client. Error: {e}")
    gpt_client = None

# ==============================================================================
#                                JUDGE LOGIC
# ==============================================================================

def get_gpt_completion(messages, json_mode=False):
    if not gpt_client:
        return "ERROR: Client not initialized"
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-5.1_2025-11-13",
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"} if json_mode else None 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling GPT: {e}")
        return "ERROR"

def llm_judge_correctness(question, ground_truth, prediction):
    if not ground_truth:
        return False, "No Ground Truth provided."

    prompt = f"""
    You are an impartial judge evaluating a Multimodal AI's response.
    
    Question: "{question}"
    Ground Truth: "{ground_truth}"
    Model Prediction: "{prediction}"
    
    Task: 
    1. Compare the factual core of the Model Prediction against the Ground Truth.
    2. Determine if the prediction matches the ground truth (ignore minor phrasing differences).
    3. Output strictly valid JSON.

    Format:
    {{
        "reason": "Short explanation...",
        "verdict": true
    }}
    """

    parity_prompt = f"""
    You are an parity checker judge evaluating a Multimodal AI's response. The model has been provided with a question but instead of a normal video, it's been given a black video. 
    
    Question: "{question}"
    Model Prediction: "{prediction}"
    
    Your task is basically to just check if the model has corretly been able to predict if the video is black or not. It can also say it's not visible, frames can be processed or anything similar along these lines.
    - If the model correctly guesses that the video is black, give verdict to be true.
    - Else, give verdict to be false. 

    You must return your response striclty in the below JSON format.
    {{
        "reason": "Short explanation...",
        "verdict": true
    }}
    """
    
    response_text = get_gpt_completion(
        [{"role": "user", "content": prompt}], 
        json_mode=True
    )
    
    try:
        data = json.loads(response_text)
        verdict = data.get("verdict", False)
        reason = data.get("reason", "No reasoning provided.")
        
        # Handle cases where model returns string "true"/"false" instead of boolean
        if isinstance(verdict, str):
            verdict = verdict.lower() == "true"
            
        return verdict, reason
        
    except json.JSONDecodeError:
        return False, f"JSON Parse Failed. Raw: {response_text}"

# ==============================================================================
#                                EXECUTION
# ==============================================================================

def run_evaluation():
    # Verify the directory exists
    if not os.path.isdir(TARGET_DIR):
        print(f"❌ Error: Directory not found: {TARGET_DIR}")
        return

    print(f"🔎 Scanning results in: {TARGET_DIR}")
    
    # Grab all JSON files in that specific folder
    result_files = glob.glob(os.path.join(TARGET_DIR, "*_results.json"))

    if not result_files:
        print("⚠️ No *_results.json files found in the directory.")
        return

    for file_path in tqdm(result_files, desc="Judging Files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            modified = False
            for entry in data:
                # Check if we need to judge
                already_judged = entry.get("judge_verdict") is not None
                
                if already_judged and not OVERWRITE_EXISTING_JUDGEMENTS:
                    continue

                # Safety check for missing ground truth
                if not entry.get("ground_truth"):
                    entry["judge_reason"] = "No Ground Truth Available"
                    entry["judge_verdict"] = False
                    modified = True
                    continue

                # Run Judge
                verdict, reason = llm_judge_correctness(
                    entry.get("question", ""), 
                    entry.get("ground_truth", ""), 
                    entry.get("model_prediction", "")
                )
                
                entry["judge_verdict"] = verdict
                entry["judge_reason"] = reason
                modified = True
            
            # Save back if changes were made
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Error processing file {file_path}: {e}")

if __name__ == "__main__":
    run_evaluation()