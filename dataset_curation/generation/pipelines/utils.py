import json
import re
from pathlib import Path

def seconds_to_hhmmss(seconds):
    # Check if the timestamp is already formatted (e.g., "00:10:33")
    if isinstance(seconds, str) and ":" in seconds:
        return seconds
        
    seconds = int(float(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def load_transcript(video_name, transcript_dir):
    """Loads and cleans SRT files."""
    srt_path = Path(transcript_dir) / f"{video_name}_transcript.srt"
    if not srt_path.exists():
        # Fallback to checking without suffix or different casing if needed
        return ""
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove timestamps and line numbers
    clean_text = re.sub(r'\n\d+\n', '\n', '\n' + content)
    clean_text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', clean_text)
    
    # Remove duplicates lines (common in SRTs) and merge
    lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
    return " ".join(lines)

def extract_metadata(agent_response):
    """Parses MMCT response for logging."""
    meta = {
        "retrieved_segments": [],
        "full_text_response": "",
        "token_usage": {},
        "video_metadata_raw": []
    }

    # --- PATH 1: Try Structured JSON (video_qna_response) ---
    if isinstance(agent_response, dict) and agent_response.get('video_qna_response'):
        qna = agent_response['video_qna_response']
        result = qna.get('result', {})
        
        meta["full_text_response"] = result.get('answer', "") or str(result)
        meta["token_usage"] = qna.get('tokens', {})
        
        if 'videos' in result:
             meta["video_metadata_raw"] = result['videos']
             for v in result['videos']:
                 # Handle list of lists structure: [[start, end], ...]
                 if 'timestamps' in v:
                     for ts in v['timestamps']:
                         meta["retrieved_segments"].append({
                             "video_id": v.get('hash_id', 'unknown'),
                             "start": ts[0],
                             "end": ts[1]
                         })

    # --- PATH 2: Fallback to Object/Content Access ---
    # This identifies the actual object inside the dictionary
    content_obj = None
    
    # FIX: Check if it's a dict with 'content' key first
    if isinstance(agent_response, dict) and 'content' in agent_response:
        content_obj = agent_response['content']
    # Legacy: Check if it's an object with 'content' attribute
    elif hasattr(agent_response, 'content'):
        content_obj = agent_response.content
    
    # Extract from the found object (VideoAgentResponse)
    if content_obj:
        # 1. Extract Text
        if hasattr(content_obj, 'response'):
            # Only overwrite if we didn't find a structured answer in Path 1
            if not meta["full_text_response"]: 
                meta["full_text_response"] = content_obj.response
        
        # 2. Extract Sources (This is where your timestamps were hiding)
        if hasattr(content_obj, 'source'):
            for src in content_obj.source:
                # MMCT sources often have a 'timestamps' list of TimestampPair objects
                if hasattr(src, 'timestamps'):
                    for ts in src.timestamps:
                        # Handle TimestampPair object (has .start_time) or tuple/list
                        start_t = ts.start_time if hasattr(ts, 'start_time') else ts[0]
                        end_t = ts.end_time if hasattr(ts, 'end_time') else ts[1]
                        
                        meta["retrieved_segments"].append({
                            "video_id": getattr(src, 'video_id', 'unknown'),
                            "start": start_t,
                            "end": end_t
                        })

    # --- Fallback: If absolutely nothing was found ---
    if not meta["full_text_response"]:
        meta["full_text_response"] = str(agent_response)

    return meta

def clean_json_string(raw_text):
    """Extracts JSON from Markdown."""
    try:
        text = raw_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text)
    except Exception as e:
        print(f"⚠️ JSON Parse Error: {e}")
        return []