import json
import os

def convert_to_llava(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    llava_data = []
    for item in data:
        # Resolve absolute path based on your specified directory
        video_path = os.path.join("/workspace/Pupil", item["video_path"])
        
        llava_entry = {
            "id": item["question_id"],
            "video": video_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<video>\n{item['question']}"
                },
                {
                    "from": "gpt",
                    "value": str(item['answer'])
                }
            ]
        }
        llava_data.append(llava_entry)

    with open(output_path, 'w') as f:
        json.dump(llava_data, f, indent=2)
    print(f"Saved {len(llava_data)} records to {output_path}")

# Run for all splits
splits = ['train', 'val', 'test']
for split in splits:
    input_file = f"/workspace/Pupil/contrastive_experiments/dataset/processed_data/{split}.json"
    output_file = f"/workspace/Pupil/contrastive_experiments/dataset/llava_data/{split}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    convert_to_llava(input_file, output_file)