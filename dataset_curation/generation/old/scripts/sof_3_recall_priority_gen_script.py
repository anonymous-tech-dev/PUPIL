import os
import json
import asyncio
from dotenv import load_dotenv
load_dotenv("/home/Pupil/dataset_curation/MMCTAgent/examples/.env")

from azure.identity import AzureCliCredential, DefaultAzureCredential, ChainedTokenCredential, get_bearer_token_provider
from openai import AzureOpenAI
from mmct.video_pipeline import VideoAgent

# --- KNOBS (Configuration) ---
TARGET_VIDEO_PATH = "/home/Pupil/dataset/videos_db/initial_v3/social_skills_mod03lec15.mp4"

OUTPUT_DIR = "/home/Pupil/dataset/queries_db/exp_v3/sof_rec_priority"
# OUTPUT_FILE = os.path.join(OUTPUT_DIR, "generated_queries.json")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(TARGET_VIDEO_PATH))[0]}_gen_queries1.json")

# Deployment Names
EVAL_MODEL = "gpt-5.1_2025-11-13" # The model used for checking uniqueness
REFINER_MODEL = "gpt-5.1_2025-11-13" # The model used for JSON correction
REFINER_API_VERSION = "2024-12-01-preview"

# Pipeline Settings
REQUIRED_QUERIES = 5
Azure_ENDPOINT = 'https://<AZURE_OPENAI_ENDPOINT>'
Azure_SCOPE = "api://azure/.default"


# --- JSON Refiner Class ---
class JSONRefiner:
    """Uses GPT-4o to clean and format messy MMCT output into valid JSON."""
    def __init__(self):
        self.deployment_name = REFINER_MODEL
        self.client = self._initialize_client()

    def _initialize_client(self):
        credential = get_bearer_token_provider(
            ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential(exclude_interactive_browser_credential=True)),
            Azure_SCOPE
        )
        return AzureOpenAI(
            azure_endpoint=Azure_ENDPOINT,
            azure_ad_token_provider=credential,
            api_version=REFINER_API_VERSION,
        )

    def refine_to_json(self, raw_text):
        prompt = f"""
        You are a JSON recovery expert. I have a messy response from a video agent that contains questions and answers.
        Extract the information and return a strictly valid JSON list of objects.
        Each object must have: "question", "answer", and "source_of_fact".

        Messy Input:
        {raw_text}

        Response format:
        [
          {{ "question": "...", "answer": "...", "source_of_fact": "video_context_priority" }}
        ]
        """
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "system", "content": "You only output valid JSON code blocks."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"} if "2024-08-06" in REFINER_MODEL else None
        )
        
        content = response.choices[0].message.content
        # Basic cleanup in case it's wrapped in Markdown
        content = content.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(content)
            # Ensure it returns a list even if GPT returns a single dict
            return data if isinstance(data, list) else data.get("questions", [data])
        except:
            print("❌ Refiner failed to produce valid JSON.")
            return []

# --- Evaluator Class ---
class GPTEvaluator:
    def __init__(self):
        self.deployment_name = EVAL_MODEL
        self.client = self._initialize_client()

    def _initialize_client(self):
        credential = get_bearer_token_provider(
            ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential(exclude_interactive_browser_credential=True)),
            Azure_SCOPE
        )
        return AzureOpenAI(
            azure_endpoint=Azure_ENDPOINT,
            azure_ad_token_provider=credential,
            api_version='2024-12-01-preview',
        )

    def get_text_only_answer(self, question):
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer based on standard knowledge."},
                {"role": "user", "content": question}
            ],
            max_completion_tokens=512,
        )
        print("llll")
        print(response.choices[0].message.content)
        print("llll")
        return response.choices[0].message.content

    # def compare_answers(self, question, video_truth, gpt_guess):
    #     prompt = f"""
    #     Question: {question}

    #     Video Ground Truth: {video_truth}
    #     Generated Answer: {gpt_guess}
        
    #     Compare the Video Ground Truth with the Generated Answer.
    #     1. Are they saying EXACTLY the same thing? -> Return FALSE.
    #     2. Does the Video Ground Truth contain specific details or nuances that the Generated Answer missed? -> Return TRUE.
        
    #     Output only TRUE or FALSE.
    #     """
    #     response = self.client.chat.completions.create(
    #         model=self.deployment_name,
    #         messages=[{"role": "user", "content": prompt}],
    #         max_completion_tokens=10,
    #     )
    #     return "TRUE" in response.choices[0].message.content.upper()

    def compare_answers(self, question, video_truth, gpt_guess):
        prompt = f"""
    You are comparing two answers.

    Question:
    {question}

    Video Ground Truth:
    {video_truth}

    Generated Answer:
    {gpt_guess}

    Determine whether the Video Ground Truth contains specific details, nuances,
    or factual elements that are missing, weakened, or absent in the Generated Answer.

    Rules:
    - If both answers convey exactly the same meaning and level of detail, decision = false
    - If the Video Ground Truth includes additional or more precise information, decision = true

    Respond ONLY in valid JSON using the following schema:
    {{
    "decision": true | false,
    "reason": "<brief explanation>"
    }}
    """

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=150,
            response_format={"type": "json_object"},  # GPT-5 compatible
        )

        # Parse JSON safely
        content = response.choices[0].message.content
        parsed = json.loads(content)

        # decision_str = "TRUE" if parsed["decision"] else "FALSE"
        # reason_str = parsed["reason"]

        return parsed["decision"], parsed["reason"]

# --- Main Processing Logic ---
async def process_video(video_path, existing_data):
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    video_name = os.path.basename(video_path)
    index_name = os.path.splitext(video_name)[0] + "_index"
    current_video_queries = existing_data.get(video_path, [])
    history_txt = "\n".join([f"- {q['question']}" for q in current_video_queries])
    
    gpt_eval = GPTEvaluator()
    json_refiner = JSONRefiner()
    new_queries = []
    
    while len(current_video_queries) + len(new_queries) < REQUIRED_QUERIES:
        needed = REQUIRED_QUERIES - (len(current_video_queries) + len(new_queries))
        print(f"🔄 Need {needed} more queries for {video_name}...")

        prompt = f"""
                You are an expert dataset curator creating a 'Context Priority' benchmark.
                Your goal is to generate {max(needed, 5)} questions where the answer derived from the video/transcript is DISTINCT from a general world-knowledge answer.

                The goal is to trap a model that relies on pre-trained knowledge instead of the video context.

                Strategies to generate these questions:
                1. **"Closed-World" Lists:** Questions like "What are the three factors affecting X mentioned in the lecture?" (General knowledge might list 5 factors; the video only lists 3 specific ones. The video answer must be the specific 3).
                2. **Specific Examples/Case Studies:** "In the example regarding [Topic], which specific compound/number/person was used?" (General knowledge provides the theory; the video provides the specific instance).
                3. **Lecturer's Definitions:** "How does the speaker define [Term] in this specific module?" (Look for simplified or non-standard definitions used for teaching purposes that differ from a rigorous textbook definition).
                4. **Constraint-Based Scenarios:** "For the specific problem shown at timestamp X, why was the solution [Y]?" (Where a general model might suggest a different standard solution, but the video forces a specific path).

                Output Format:
                Return a list of questions and their strict video-based answers.

                Examples:
                (BAD): "What is the atomic weight of Carbon?" (Universal fact - video and external knowledge are identical).
                (GOOD): "In the lecture's example of calculating molar mass, which specific rounding value was used for Carbon?" (Video might use 12.0, while precise external knowledge is 12.011 - a perfect test of priority).

                Already Accepted Questions (Do not repeat):
                {history_txt}
                """
                
        video_agent = VideoAgent(query=prompt, index_name=index_name, use_critic_agent=False)

        try:
            print(f"🤖 Querying MMCT Agent...")
            agent_raw_response = await video_agent()
            print("ccc")
            print(agent_raw_response)
            print("ccc")
            
            # Extract content string safely from MMCT response object
            raw_text = str(agent_raw_response)
            if hasattr(agent_raw_response, 'content'):
                raw_text = str(agent_raw_response.content)

            print("🧹 Refining output with GPT-4o...")
            candidates = json_refiner.refine_to_json(raw_text)
            print("ppp")
            print(candidates)
            print("ppp")

            if not candidates:
                continue

            for item in candidates:
                if len(current_video_queries) + len(new_queries) >= REQUIRED_QUERIES:
                    break
                
                q_text = item.get("question")
                v_ans = item.get("answer")
                if not q_text or not v_ans: continue

                print(f"❓ Testing: {q_text[:60]}...")
                gpt_ans = gpt_eval.get_text_only_answer(q_text)
                print("aa")
                print(gpt_ans)
                print("aa")

                decision, decision_reason = gpt_eval.compare_answers(q_text, v_ans, gpt_ans)
                if decision:
                    print("✅ Kept: Video unique.")
                    item["source_of_fact"] = "video_specific_definition"
                    item["decision_reason"] = decision_reason
                    new_queries.append(item)
                    history_txt += f"\n- {q_text}"
                else:
                    print("❌ Discarded: Too generic: ", decision_reason)
                    
        except Exception as e:
            print(f"⚠️ Error in loop: {e}")
            await asyncio.sleep(2) # Backoff

    # Save logic
    if new_queries:
        existing_data.setdefault(video_path, []).extend(new_queries)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(existing_data, f, indent=4)
        print(f"💾 Saved {len(new_queries)} new queries.")

if __name__ == "__main__":
    db = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            try: db = json.load(f)
            except: db = {}
    
    asyncio.run(process_video(TARGET_VIDEO_PATH, db))