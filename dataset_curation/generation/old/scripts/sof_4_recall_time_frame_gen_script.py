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
OUTPUT_DIR = "/home/Pupil/dataset/queries_db/exp_v3/sof_rec_time"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(TARGET_VIDEO_PATH))[0]}_gen_queries.json")

# Deployment & API Versions
VALIDATOR_MODEL = "gpt-5.1_2025-11-13"   # The logic checker
REFINER_MODEL = "gpt-5.1_2025-11-13"      # The JSON formatter
REFINER_API_VERSION = "2024-12-01-preview"
# Azure_ENDPOINT = 'https://<AZURE_OPENAI_ENDPOINT>/'
Azure_ENDPOINT = 'https://<AZURE_OPENAI_ENDPOINT>'
Azure_SCOPE = "api://azure/.default"

# Pipeline Settings
REQUIRED_QUERIES = 5
MMCT_TIMEOUT = 120.0


# --- JSON Refiner Class ---
class JSONRefiner:
    """Uses GPT-4o to extract and format valid JSON from messy MMCT output."""
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
        You are a JSON formatting assistant. I have a response from a video agent.
        Extract the questions and answers into a valid JSON list. 
        Each object must have "question", "answer", and "source_of_fact".

        Raw Input:
        {raw_text}

        Response format:
        [
          {{ "question": "...", "answer": "...", "source_of_fact": "temporal_synthesis" }}
        ]
        """
        # try:
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You output strictly valid JSON lists only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"} if "2024-08-06" in REFINER_MODEL else None
        )
        content = response.choices[0].message.content.strip()
        print("bbbbbbbbbb")
        print(content)
        print("bbbbbbbbbb")
        # Remove potential markdown wrapping
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        # Ensure output is a list
        return data if isinstance(data, list) else data.get("questions", [data])
        # except Exception as e:
        #     print(f"❌ Refiner Error: {e}")
        #     return []

# --- Validator Class ---
class TemporalValidator:
    def __init__(self):
        self.deployment_name = VALIDATOR_MODEL
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

    # def validate_temporal_nature(self, question):
    #     prompt = f"""
    #     Question: "{question}"
    #     Does answering this strictly require synthesizing information from at least two separate moments in a video?
    #     (e.g., comparing start vs end, aggregating details shown over time).
    #     Output YES or NO only.
    #     """
    #     response = self.client.chat.completions.create(
    #         model=self.deployment_name,
    #         messages=[{"role": "user", "content": prompt}],
    #         max_completion_tokens=100,
    #     )
    #     return "YES" in response.choices[0].message.content.upper()

    def validate_temporal_nature(self, question):
        prompt = f"""
        Question: "{question}"
        Does answering this strictly require synthesizing information from at least two separate moments in a video?
        (e.g., comparing start vs end, aggregating details shown over time).
        
        Output strictly in this JSON format:
        {{
            "verdict": "YES" or "NO",
            "reason": "Brief explanation of why this requires (or doesn't require) temporal synthesis."
        }}
        """
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a logic validator. Output strictly valid JSON."},
                {"role": "user", "content": prompt}
            ],
            # Ensure JSON mode is used if model supports it, otherwise text is fine given the prompt
            response_format={"type": "json_object"} if "gpt-4o" in self.deployment_name or "2024" in self.deployment_name else None
        )
        
        try:
            content = response.choices[0].message.content.strip()
            # Clean markdown if present
            content = content.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(content)
            is_valid = "YES" in data.get("verdict", "").upper()
            reason = data.get("reason", "No reason provided")
            
            return is_valid, reason
            
        except Exception as e:
            print(f"⚠️ Validation Parsing Error: {e}")
            # Default to False if we can't parse the reason
            return False, "Error parsing validator response"

async def process_video(video_path, existing_data):
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    video_name = os.path.basename(video_path)
    index_name = os.path.splitext(video_name)[0] + "_index"
    current_video_queries = existing_data.get(video_path, [])
    history_txt = "\n".join([f"- {q['question']}" for q in current_video_queries])
    
    validator = TemporalValidator()
    json_refiner = JSONRefiner()
    new_queries = []
    
    while len(current_video_queries) + len(new_queries) < REQUIRED_QUERIES:
        needed = REQUIRED_QUERIES - (len(current_video_queries) + len(new_queries))
        print(f"\n🔄 Need {needed} more queries for {video_name}...")

        print("histry")
        if history_txt == "":
            history_txt = "none"
        print(history_txt)
        print("histry")
        prompt = f"""
            You are an expert examiner. Generate ANY {max(needed, 5)} "Temporal Synthesis" questions along with their respective answers.
            
            CRITICAL REQUIREMENT:
            Each question MUST require the viewer to bridge information from at least two distinct distinct moments in the video to answer.
            
            Use these "Bridge Types" to find questions:
            1. **Evolution**: How does [Concept X] introduced at the start evolve or change by the end?
            2. **Causality**: How does the [Action/Setup] shown early on explain the [Result/Error] seen later?
            3. **Contrast**: Compare the [Theory/Definition] explained in the first half with the [Practical Example] shown in the second half.
            4. **Step-Skipping**: "To get from Step 1 to Step 5, what crucial intermediate step was emphasized?"
            
            BAD Question: "What is said at 02:00?" (Single timestamp query)
            GOOD Question: "How does the safety warning given during the setup explain the accident shown in the final experiment?"
            
            Use descriptive references (e.g., "the initial setup", "the final diagram", timestamps (like "at 05:00")).

            Follow the below format:

            Q1: ...
            A1: ...

            Q2: ...
            A2: ...

            and so on...

            Already Accepted (DO NOT REPEAT): 
            {history_txt}
        """

        video_agent = VideoAgent(query=prompt, index_name=index_name, use_critic_agent=False)

        # try:
        print(f"🤖 Querying MMCT Agent...")
        agent_response = await asyncio.wait_for(video_agent(), timeout=MMCT_TIMEOUT)

        print("aaaa")
        print(agent_response)
        print("aaaa")
        
        # Extract text content safely
        raw_text = str(agent_response)
        if hasattr(agent_response, 'content'):
            raw_text = str(agent_response.content.response if hasattr(agent_response.content, 'response') else agent_response.content)

        print("🧹 Refining JSON with GPT-4o...")
        candidates = json_refiner.refine_to_json(raw_text)

        if not candidates: continue

        for item in candidates:
            if len(current_video_queries) + len(new_queries) >= REQUIRED_QUERIES:
                break
            
            q_text = item.get("question")
            if not q_text: continue

            print(f"❓ Testing Temporal Logic: {q_text[:70]}...")
            is_valid, decision_reason = validator.validate_temporal_nature(q_text)

            if is_valid:
                print("✅ Kept: Valid temporal query.")
                item["decision_reason"] = decision_reason
                new_queries.append(item)
                history_txt += f"\n- {q_text}"
            else:
                print("❌ Discarded: Single timestamp: ", decision_reason)

        # except asyncio.TimeoutError:
        #     print("⚠️ TIMEOUT: MMCT Agent hung. Retrying...")
        # except Exception as e:
        #     print(f"⚠️ Loop Error: {e}")

    if new_queries:
        existing_data.setdefault(video_path, []).extend(new_queries)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(existing_data, f, indent=4)
        print(f"💾 Saved {len(new_queries)} new queries for {video_name}")

if __name__ == "__main__":
    db = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            try: db = json.load(f)
            except: db = {}
    asyncio.run(process_video(TARGET_VIDEO_PATH, db))
