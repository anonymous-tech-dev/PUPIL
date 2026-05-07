import os
import json
import glob
from tqdm import tqdm
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

# ==============================================================================
#                                KNOBS & PATHS
# ==============================================================================

# 1. Base paths
BASE_OUTPUTS_DIR = "/workspace/Pupil/contrastive_experiments/outputs"
DATASET_PATH = "/workspace/Pupil/contrastive_experiments/dataset/llava_data/test.json"

# 2. Runs to evaluate
RUNS_TO_EVALUATE = [
    # "baseline_qwen3vl_8b_8nf",
    # "baseline_qwen3vl_8b_16nf",
    # "baseline_qwen3vl_8b_32nf"
    "contrastive_sft_v01_run1_batch_f8_bs32",
    "contrastive_sft_v01_run3_batch_f16_bs32",
    "contrastive_sft_v02_run1_blackened_f8_bs32",
    "contrastive_sft_v02_run3_blackened_f16_bs32",
    "contrastive_sft_v03_run1_gaussian_f8_bs32",
    "contrastive_sft_v03_run3_gaussian_f16_bs32",
]

# 3. Save interval (how often to write to JSON to ensure hot-resumability)
SAVE_INTERVAL = 5 

# ==============================================================================
#                               AZURE SETUP
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
        return '{"verdict": "no", "reason": "ERROR: Client not initialized"}'
    
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
        return '{"verdict": "no", "reason": "API Error"}'

def llm_judge_correctness(question, ground_truth, prediction):
    if not ground_truth:
        return "no", "No Ground Truth provided."

    prompt = f"""
    You are an impartial judge evaluating a vision-language model's answer to a video-related question (ActivityNet domain).
    
    Question: "{question}"
    Ground Truth: "{ground_truth}"
    Model Prediction: "{prediction}"
    
    Task: 
    1. Compare the factual core of the Model Prediction against the Ground Truth Answer.
    2. Determine if the prediction correctly answers the question according to the ground truth. 
       Ignore minor grammatical differences, extra context, or phrasing changes, as long as the core truth is captured.
    3. Output strictly valid JSON.
    
    Format:
    {{
        "reason": "Short explanation of why it matches or fails...",
        "verdict": "yes" or "no"
    }}
    """
    
    response_text = get_gpt_completion(
        [{"role": "user", "content": prompt}], 
        json_mode=True
    )
    
    try:
        data = json.loads(response_text)
        verdict = str(data.get("verdict", "no")).strip().lower()
        reason = data.get("reason", "No reasoning provided.")
        
        # Cleanup edge cases where model might output true/false instead of yes/no
        if verdict == "true": verdict = "yes"
        if verdict == "false": verdict = "no"
            
        return verdict, reason
        
    except json.JSONDecodeError:
        return "no", f"JSON Parse Failed. Raw: {response_text}"

# ==============================================================================
#                                EXECUTION
# ==============================================================================

def load_dataset_questions(dataset_path):
    """Loads the test.json and extracts questions sequentially."""
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    questions = []
    for item in data:
        convs = item.get("conversations", [])
        for c in convs:
            if c.get("from") == "human":
                # Remove the <video> tag and any leading/trailing whitespace
                q = c.get("value", "").replace("<video>\n", "").replace("<video>", "").strip()
                questions.append(q)
                break # Only take the first human prompt
    return questions

def find_evaluation_file(run_dir):
    """Checks typical locations for evaluation_results.json in a run directory."""
    possible_paths = [
        os.path.join(run_dir, "evaluation_results.json"),
        os.path.join(run_dir, "test_results", "evaluation_results.json"),
        os.path.join(run_dir, "test_results.json")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None

def run_evaluation():
    print("Loading Original Questions from Dataset...")
    questions = load_dataset_questions(DATASET_PATH)
    if not questions:
        return
    print(f"Loaded {len(questions)} questions from test dataset.")

    for run_name in RUNS_TO_EVALUATE:
        run_dir = os.path.join(BASE_OUTPUTS_DIR, run_name)
        file_path = find_evaluation_file(run_dir)
        
        if not file_path:
            print(f"⚠️ Skipping {run_name}: Could not find evaluation JSON.")
            continue
            
        print(f"\n🔎 Processing Run: {run_name}")
        print(f"📁 File: {file_path}")
        
        # Load the target evaluation file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        predictions = data.get("predictions", [])
        references = data.get("references", [])
        
        # Initialize single array for evaluations if it doesn't exist
        if "llm_judge_evaluations" not in data:
            data["llm_judge_evaluations"] = []
            
        start_idx = len(data["llm_judge_evaluations"])
        total_items = len(predictions)
        
        if start_idx >= total_items:
            print(f"✅ Run {run_name} is already fully evaluated. Skipping.")
            continue
            
        if len(predictions) != len(questions):
            print(f"⚠️ Warning: Dataset has {len(questions)} questions but run has {len(predictions)} predictions. Ensure indices match.")
            
        print(f"🔄 Resuming from index {start_idx} / {total_items}")
        
        modified = False
        
        # Process remaining items
        for i in tqdm(range(start_idx, total_items), desc=f"Evaluating {run_name}"):
            q = questions[i] if i < len(questions) else "Unknown Question"
            pred = predictions[i]
            ref = references[i]
            
            verdict, reason = llm_judge_correctness(q, ref, pred)
            
            # Append as a single dictionary object
            data["llm_judge_evaluations"].append({
                "verdict": verdict,
                "reason": reason
            })
            modified = True
            
            # Hot-resumable save step
            if (i + 1) % SAVE_INTERVAL == 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
        # Final save and Metrics calculation
        if modified:
            # Calculate final score
            evaluations = data["llm_judge_evaluations"]
            yes_count = sum(1 for e in evaluations if e.get("verdict") == "yes")
            score = yes_count / len(evaluations) if evaluations else 0.0
            
            # Inject score into the metrics dictionary
            if "metrics" not in data:
                data["metrics"] = {}
            data["metrics"]["llm_judge_score"] = score
            
            # Final write
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            print(f"🏁 Finished {run_name} | LLM Judge Score: {score:.4f}")

if __name__ == "__main__":
    run_evaluation()