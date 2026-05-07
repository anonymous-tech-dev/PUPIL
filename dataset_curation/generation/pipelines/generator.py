import asyncio
import json
import random
from pathlib import Path
from models.wrappers import ModelManager
import prompts
from config import GPT_EVAL_MODEL, GPT_REFINER_MODEL,VIDEO_CATEGORY_MAP, DEFAULT_CATEGORY_MIX, MAX_ATTEMPTS
from .utils import extract_metadata, clean_json_string, load_transcript, seconds_to_hhmmss

class BenchmarkGenerator:
    def __init__(self, video_name, video_path, transcript_path, output_dir):
        self.video_name = video_name
        self.video_path = video_path
        self.transcript_text = load_transcript(video_name, transcript_path)
        self.output_dir = Path(output_dir)
        self.models = ModelManager()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.query_file = self.output_dir / f"{video_name}_queries.json"
        self.meta_file = self.output_dir / f"{video_name}_metadata.jsonl"

    def load_existing_data(self):
        if self.query_file.exists():
            with open(self.query_file, 'r') as f:
                data = json.load(f)
                # Handle dictionary wrapper if present
                if isinstance(data, dict):
                    return list(data.values())[0] if data else []
                return data
        return []

    def save_data(self, queries, metadata_entry):
        # Save readable JSON
        final_struct = {str(self.video_path): queries}
        with open(self.query_file, 'w') as f:
            json.dump(final_struct, f, indent=2)
        
        # Append heavy metadata
        with open(self.meta_file, 'a') as f:
            f.write(json.dumps(metadata_entry) + "\n")

    # --- VALIDATORS ---
    
    async def validate_priority(self, question, answer):
        """Pass if video answer is DISTINCT from general knowledge."""
        blind_ans = self.models.get_gpt_completion(
            [{"role": "user", "content": f"Query: {question}"}],
            model=GPT_EVAL_MODEL
        )
        comparison = self.models.get_gpt_completion(
            [{"role": "user", "content": prompts.COMPARE_ANSWERS_STRICT.format(
                ground_truth=answer, prediction=blind_ans)}],
            model=GPT_EVAL_MODEL
        )
        
        # Capture logs
        logs = {
            "check_type": "priority",
            "blind_answer": blind_ans,
            "comparison_response": comparison
        }

        if "DIFFERENT" in comparison:
            return True, f"Unique to video. Blind guess: {blind_ans}", logs
        return False, "Too generic / Answerable by pre-training.", logs

    async def validate_temporal(self, question, answer):
        """Pass if question requires bridging time."""
        logs = {"check_type": "temporal"}

        # 1. Blind Knowledge Check
        blind_ans = self.models.get_gpt_completion(
            [{"role": "user", "content": prompts.ANSWER_BLINDLY.format(question=question)}],
            model=GPT_EVAL_MODEL
        )
        is_known = self.models.get_gpt_completion(
            [{"role": "user", "content": prompts.JUDGE_CORRECTNESS.format(
                question=question, ground_truth=answer, prediction=blind_ans)}],
            model=GPT_EVAL_MODEL
        )
        
        logs["blind_answer"] = blind_ans
        logs["is_known_judge"] = is_known

        if "YES" in is_known:
            print("   ❌ Rejected: Known from pre-training.")
            return False, "Rejected, Known from pre-training.", logs
        else:
            print("   ✅ GPT couldn't answer from pre-training.")

        resp = self.models.get_gpt_completion(
            [{"role": "user", "content": prompts.VALIDATE_TEMPORAL.format(question=question)}],
            model=GPT_EVAL_MODEL,
            json_mode=True
        )
        logs["temporal_check_json"] = resp

        try:
            data = json.loads(resp)
            if "YES" in data.get("verdict", "").upper():
                return True, data.get("reason"), logs
        except: pass
        return False, "Does not require temporal synthesis.", logs

    async def validate_query(self, question, answer, sof_type):
        """Runs the Blind check and Transcript check logic."""
        logs = {"check_type": sof_type}
        
        # 1. Blind Knowledge Check
        blind_ans = self.models.get_gpt_completion(
            [{"role": "user", "content": prompts.ANSWER_BLINDLY.format(question=question)}],
            model=GPT_EVAL_MODEL
        )
        is_known = self.models.get_gpt_completion(
            [{"role": "user", "content": prompts.JUDGE_CORRECTNESS.format(
                question=question, ground_truth=answer, prediction=blind_ans)}],
            model=GPT_EVAL_MODEL
        )
        
        logs["blind_answer"] = blind_ans
        logs["is_known_judge"] = is_known

        if "YES" in is_known:
            print("   ❌ Rejected: Known from pre-training.")
            return False, "Known from pre-training", logs
        else:
            print("   ✅ GPT couldn't answer from pre-training.")

        # 2. Transcript Check
        if not self.transcript_text:
            return True, "No transcript available (Skipped check)", logs
            
        trans_ans = self.models.get_gpt_completion(
            [{"role": "user", "content": prompts.ANSWER_FROM_TRANSCRIPT.format(
                transcript_snippet=self.transcript_text[:15000], question=question)}],
            model=GPT_EVAL_MODEL
        )
        in_transcript = self.models.get_gpt_completion(
            [{"role": "user", "content": prompts.JUDGE_CORRECTNESS.format(
                question=question, ground_truth=answer, prediction=trans_ans)}],
            model=GPT_EVAL_MODEL
        )
        
        logs["transcript_answer"] = trans_ans
        logs["in_transcript_judge"] = in_transcript

        if sof_type == 'audio':
            if "YES" not in in_transcript:
                print("   ❌ Rejected: SoF is Audio but answer not in transcript.")
                return False, "Audio Q but answer not in transcript", logs
            else:
                print("   ✅ SoF is Audio and GPT could answer with transcript.")
        elif sof_type == 'visual':
            if "YES" in in_transcript:
                print("   ❌ Rejected: SoF is Visual but answer found in transcript.")
                return False, "Visual Q but answer found in transcript", logs
            else:
                print("   ✅ SoF is Video and GPT couldn't answer with transcript.")
        
        return True, "Passed all checks", logs

    async def classify_category(self, question, answer, sof_type):
        """Categorize into the new 5 buckets defined in config."""
        
        # Use the centralized prompt from prompts.py
        formatted_prompt = prompts.CLASSIFY_QUERY.format(
            question=question, 
            answer=answer, 
            sof_type=sof_type
        )
        
        resp = self.models.get_gpt_completion(
             [{"role": "user", "content": formatted_prompt}], 
             model=GPT_EVAL_MODEL, 
             json_mode=True
        )
        
        try: 
            return json.loads(resp)
        except Exception as e:
            print(f"⚠️ Classification Parse Error: {e}")
            return {"category": "Unclassified", "reasoning": "JSON Parse Error"}

    # --- MAIN PIPELINE ---

    async def run_pipeline(self, pipeline_mode, target_count, category_nudge=None):
        print(f"🚀 Pipeline: {pipeline_mode.upper()} | Nudge: {category_nudge}")
        current_queries = self.load_existing_data()
        index_name = f"{self.video_name}_index"
        

        # -------- check in case the ingestion is faulty to skip source of fact -----------
        attempts = 0
        # MAX_ATTEMPTS = 25
        
        while len(current_queries) < target_count:
            if attempts >= MAX_ATTEMPTS:
                print(f"⚠️ Max attempts ({MAX_ATTEMPTS}) reached for {pipeline_mode}. Skipping to next...")
                break
            
            attempts += 1

            needed = target_count - len(current_queries)
            print(f"\n🔄 Generating batch for {self.video_name}... Need {needed} more (Attempt {attempts}/{MAX_ATTEMPTS})")
            
            history_txt = "\n".join([f"- {q['question']}" for q in current_queries])
            
            # Select Base Prompt based on Pipeline Mode
            if pipeline_mode == 'priority':
                base_prompt = prompts.GENERATE_PRIORITY_QUESTIONS
            elif pipeline_mode == 'time':
                base_prompt = prompts.GENERATE_TIME_QUESTIONS
            elif pipeline_mode == 'visual':
                base_prompt = prompts.GENERATE_QUESTIONS_VISUAL
            elif pipeline_mode == 'audio':
                base_prompt = prompts.GENERATE_QUESTIONS_AUDIO
            else:
                print("HUH")
                break
            
            # 1. Determine the "Mix" for this specific video
            # Check if any key in the map is a substring of the video name
            # e.g. if video_name is "nptel_robotics_lec01", it matches "robotics"
            active_mix = DEFAULT_CATEGORY_MIX
            for key, mix in VIDEO_CATEGORY_MAP.items():
                if key in self.video_name:
                    active_mix = mix
                    break
            
            # 2. Select the category based on current count
            # This ensures if we have 0 qs, we keep trying for category[0] until we succeed.
            # target_index = len(current_queries) % len(active_mix)
            # current_cat_key = active_mix[target_index]

            # randomly pick an entry from the corresponding key from config
            current_cat_key = random.choice(active_mix)
            
            # 3. Get the prompt text
            current_nudge_text = prompts.CATEGORY_INSTRUCTIONS.get(current_cat_key, "")
            
            print(f"  👉 Target Category: {current_cat_key} (for Q#{len(current_queries)+1})")

            full_prompt = base_prompt.format(
                topic="Education",
                num_needed=min(needed + 2, 1), # Ask for a few candidates
                category_nudge=current_nudge_text,
                history=history_txt
            )

            # MMCT Call
            agent = self.models.get_video_agent(index_name, full_prompt)
            try:
                raw_response = await agent()
            except Exception as e:
                print(f"⚠️ MMCT Error: {e}")
                continue
                
            # Process
            metadata = extract_metadata(raw_response)
            
            refiner_prompt = prompts.REFINE_TO_JSON.format(raw_text=metadata["full_text_response"])
            refined_json = self.models.get_gpt_completion(
                [{"role": "user", "content": refiner_prompt}], model=GPT_REFINER_MODEL
            )
            candidates = clean_json_string(refined_json)

            if not candidates: continue

            # Validation Loop
            for item in candidates:
                if len(current_queries) >= target_count: break
                
                q, a = item.get('question'), item.get('answer')
                if not q or not a: continue
                if any(x['question'] == q for x in current_queries): continue

                print(f"🔎 Validating: {q[:60]}...")
                
                valid = False
                reason = ""

                # Logic Switch
                valid = False
                reason = ""
                validation_logs = {}

                if pipeline_mode == 'priority':
                    valid, reason, validation_logs = await self.validate_priority(q, a)
                elif pipeline_mode == 'time':
                    valid, reason, validation_logs = await self.validate_temporal(q, a)
                else:
                    # Updated to catch the tuple return
                    valid, reason, validation_logs = await self.validate_query(q, a, pipeline_mode)

                if valid:
                    # Classify
                    cat_info = await self.classify_category(q, a, pipeline_mode)
                    
                    formatted_segments = [
                        {
                            "start": seconds_to_hhmmss(seg["start"]),
                            "end": seconds_to_hhmmss(seg["end"])
                        }
                        for seg in metadata.get("retrieved_segments", [])
                    ]

                    final_item = {
                        "query_id": f"{self.video_name}_{pipeline_mode}_{len(current_queries)+1:03d}",
                        "question": q,
                        "ground_truth": a,
                        "annotations": {
                            "pipeline_mode": pipeline_mode,
                            "cognitive_category": cat_info.get("category"),
                            "reasoning": cat_info.get("reasoning")
                        },
                        "timestamp_segments": formatted_segments
                    }
                    current_queries.append(final_item)
                    
                    # Add the raw validation logs to the metadata (saved to .jsonl)
                    metadata["linked_query_id"] = final_item["query_id"]
                    metadata["validation_checks"] = validation_logs  
                    metadata["generation_nudge"] = current_cat_key      
                    metadata["validation_reason"] = reason              
                    metadata["attempt_number"] = attempts               
                    
                    self.save_data(current_queries, metadata)
                    print(f"   ✅ Accepted! [{cat_info.get('category')}]")
                else:
                    print(f"   ❌ Rejected: {reason}")