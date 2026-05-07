#!/usr/bin/env env python3
"""
Parity Video Evaluator
Evaluates Qwen3VL's ability to detect black/empty video frames and classifies responses using GPT-5
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import torch

# ==================== CONFIGURATION KNOBS ====================
# Paths
QUERIES_JSON_PATH = "/home/Pupil/dataset/queries_db/parity_queries/temp_comb.json"
OUTPUT_JSON_PATH = "/home/Pupil/dataset/queries_db/parity_queries/final_parity_results.json"
VIDEO_BASE_PATH = "/home/Pupil/dataset/videos_db"

# Behavior flags
DYNAMIC_DELETE_VARIABLE = False  # If True, removes entries from source JSON when Qwen correctly identifies black video
SKIP_EXISTING_EVALUATIONS = True  # If True, skips questions already present in output JSON

# Model settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# GPT-5 Classification prompt
GPT5_CLASSIFICATION_PROMPT = """You are analyzing a response from a video understanding AI model.

The model was shown a completely BLACK video (no visible content, just black frames) and asked a question about visual details.

The model's response was:
"{response}"

Your task: Determine if the model CORRECTLY identified that the video was black/empty/had no visible content, OR if it hallucinated and answered as if it could see content.

Respond with EXACTLY ONE of these classifications:
- CORRECT_BLACK_DETECTION: The model clearly stated it cannot see anything, video is black, no content visible, etc.
- HALLUCINATION: The model described visual content, answered the question as if it could see things, or provided specific details
- AMBIGUOUS: Unclear or hedged response

Return ONLY the classification label, nothing else."""

# ==============================================================


class ParityVideoEvaluator:
    """Orchestrates Qwen3VL evaluation and GPT-5 classification"""
    
    def __init__(self):
        self.queries_data = None
        self.output_data = {}
        self.qwen_evaluator = None
        self.gpt_evaluator = None
        
    def load_queries(self):
        """Load the queries JSON file"""
        print(f"📂 Loading queries from: {QUERIES_JSON_PATH}")
        with open(QUERIES_JSON_PATH, 'r') as f:
            self.queries_data = json.load(f)
        print(f"✅ Loaded {len(self.queries_data)} video entries")
    
    def load_existing_results(self):
        """Load existing output JSON if it exists"""
        if os.path.exists(OUTPUT_JSON_PATH):
            print(f"📂 Loading existing results from: {OUTPUT_JSON_PATH}")
            with open(OUTPUT_JSON_PATH, 'r') as f:
                self.output_data = json.load(f)
            print(f"✅ Loaded {sum(len(v) for v in self.output_data.values())} existing evaluations")
        else:
            print("📝 No existing results found, starting fresh")
            self.output_data = {}
    
    def save_results(self):
        """Save results to output JSON"""
        print(f"💾 Saving results to: {OUTPUT_JSON_PATH}")
        os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
        with open(OUTPUT_JSON_PATH, 'w') as f:
            json.dump(self.output_data, f, indent=4)
        print("✅ Results saved")
    
    def save_queries(self):
        """Save updated queries JSON (after dynamic deletion)"""
        if DYNAMIC_DELETE_VARIABLE:
            print(f"💾 Saving updated queries to: {QUERIES_JSON_PATH}")
            with open(QUERIES_JSON_PATH, 'w') as f:
                json.dump(self.queries_data, f, indent=4)
            print("✅ Queries updated")
    
    def get_black_video_path(self, original_video_path: str) -> str:
        """
        Convert original video path to black video path
        Example: /path/to/videos_db/inital_v2/video.mp4 -> /path/to/videos_db/inital_v2/parity/video.mp4
        """
        path_obj = Path(original_video_path)
        video_name = path_obj.name
        
        # Construct parity path
        parity_path = path_obj.parent / "parity" / video_name
        
        return str(parity_path)
    
    def is_already_evaluated(self, video_path: str, question: str) -> bool:
        """Check if this question has already been evaluated"""
        if not SKIP_EXISTING_EVALUATIONS:
            return False
        
        if video_path not in self.output_data:
            return False
        
        for entry in self.output_data[video_path]:
            if entry.get('question') == question:
                return True
        
        return False
    
    def initialize_models(self):
        """Initialize Qwen3VL and GPT-5 evaluators"""
        print("🤖 Initializing Qwen3VL evaluator...")
        sys.path.append('/home/Pupil')
        from models.qwen_3_vl import Qwen3VLEvaluator
        
        self.qwen_evaluator = Qwen3VLEvaluator(device=DEVICE, dtype=DTYPE)
        self.qwen_evaluator.load()
        print("✅ Qwen3VL loaded")
        
        print("🤖 Initializing GPT-5 evaluator...")
        from models.gpt import GPTAzureEvaluator
        
        self.gpt_evaluator = GPTAzureEvaluator()
        self.gpt_evaluator.load()
        print("✅ GPT-5 loaded")
    
    def evaluate_single_query(self, original_video_path: str, question: str, answer: str) -> Dict[str, Any]:
        """Evaluate a single query with Qwen3VL and classify with GPT-5"""
        
        # Get black video path
        black_video_path = self.get_black_video_path(original_video_path)
        
        if not os.path.exists(black_video_path):
            print(f"⚠️  Black video not found: {black_video_path}")
            return {
                'question': question,
                'expected_answer': answer,
                'qwen_response': 'ERROR: Black video file not found',
                'gpt5_classification': 'ERROR',
                'black_video_path': black_video_path,
                'error': 'Video file not found'
            }
        
        # Get Qwen's response
        print(f"  🔍 Querying Qwen3VL...")
        try:
            qwen_response = self.qwen_evaluator.generate_response(black_video_path, question)
        except Exception as e:
            print(f"  ❌ Qwen error: {e}")
            return {
                'question': question,
                'expected_answer': answer,
                'qwen_response': f'ERROR: {str(e)}',
                'gpt5_classification': 'ERROR',
                'black_video_path': black_video_path,
                'error': str(e)
            }
        
        print(f"  💬 Qwen response: {qwen_response[:100]}...")
        
        # Classify with GPT-5
        print(f"  🔍 Classifying with GPT-5...")
        gpt5_prompt = GPT5_CLASSIFICATION_PROMPT.format(response=qwen_response)
        
        try:
            # Use a dummy black video for GPT-5 (it only needs text)
            gpt5_classification = self.gpt_evaluator.generate_response(
                black_video_path, 
                gpt5_prompt
            ).strip()
        except Exception as e:
            print(f"  ❌ GPT-5 error: {e}")
            gpt5_classification = f"ERROR: {str(e)}"
        
        print(f"  ✅ Classification: {gpt5_classification}")
        
        return {
            'question': question,
            'expected_answer': answer,
            'qwen_response': qwen_response,
            'gpt5_classification': gpt5_classification,
            'black_video_path': black_video_path
        }
    
    def run_evaluation(self):
        """Main evaluation loop"""
        
        total_queries = sum(len(queries) for queries in self.queries_data.values())
        processed = 0
        skipped = 0
        correct_detections = 0
        
        videos_to_delete = []  # Track which video entries should be deleted
        
        for original_video_path, queries in self.queries_data.items():
            print(f"\n{'='*80}")
            print(f"📹 Processing video: {Path(original_video_path).name}")
            print(f"{'='*80}")
            
            # Initialize output entry if needed
            if original_video_path not in self.output_data:
                self.output_data[original_video_path] = []
            
            queries_to_delete = []  # Track which queries to delete from this video
            
            for query_entry in queries:
                question = query_entry['question']
                answer = query_entry['answer']
                
                # Check if already evaluated
                if self.is_already_evaluated(original_video_path, question):
                    print(f"⏭️  Skipping (already evaluated): {question[:60]}...")
                    skipped += 1
                    continue
                
                print(f"\n📝 Question: {question[:80]}...")
                
                # Evaluate
                result = self.evaluate_single_query(original_video_path, question, answer)
                
                # Store result
                self.output_data[original_video_path].append(result)
                processed += 1
                
                # Check if Qwen correctly detected black video
                if result['gpt5_classification'] == 'CORRECT_BLACK_DETECTION':
                    correct_detections += 1
                    if DYNAMIC_DELETE_VARIABLE:
                        queries_to_delete.append(query_entry)
                        print(f"  🗑️  Marking for deletion (correct detection)")
                
                # Save intermediate results every 5 queries
                if processed % 5 == 0:
                    self.save_results()
            
            # Delete queries from this video if needed
            if DYNAMIC_DELETE_VARIABLE and queries_to_delete:
                for query_to_delete in queries_to_delete:
                    self.queries_data[original_video_path].remove(query_to_delete)
                
                # If all queries deleted, mark video for deletion
                if len(self.queries_data[original_video_path]) == 0:
                    videos_to_delete.append(original_video_path)
        
        # Delete empty video entries
        if DYNAMIC_DELETE_VARIABLE:
            for video_path in videos_to_delete:
                del self.queries_data[video_path]
        
        # Final save
        self.save_results()
        if DYNAMIC_DELETE_VARIABLE:
            self.save_queries()
        
        # Summary
        print(f"\n{'='*80}")
        print(f"📊 EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total queries: {total_queries}")
        print(f"Processed: {processed}")
        print(f"Skipped (already evaluated): {skipped}")
        print(f"Correct black detections: {correct_detections}/{processed} ({100*correct_detections/max(1,processed):.1f}%)")
        if DYNAMIC_DELETE_VARIABLE:
            print(f"Queries deleted from source: {correct_detections}")
        print(f"{'='*80}")


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("🎬 PARITY VIDEO EVALUATOR")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Queries JSON: {QUERIES_JSON_PATH}")
    print(f"  - Output JSON: {OUTPUT_JSON_PATH}")
    print(f"  - Dynamic deletion: {DYNAMIC_DELETE_VARIABLE}")
    print(f"  - Skip existing: {SKIP_EXISTING_EVALUATIONS}")
    print(f"  - Device: {DEVICE}")
    print("="*80 + "\n")
    
    evaluator = ParityVideoEvaluator()
    
    # Load data
    evaluator.load_queries()
    evaluator.load_existing_results()
    
    # Initialize models
    evaluator.initialize_models()
    
    # Run evaluation
    evaluator.run_evaluation()
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()