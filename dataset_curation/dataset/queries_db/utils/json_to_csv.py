import json
import pandas as pd

def convert_json_to_excel(json_filepath, output_filepath):
    # 1. Load the JSON data
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    rows = []

    # 2. Loop through the JSON
    for video_key, items in data.items():
        # Clean up the video name to match your desired output (removes .mp4)
        video_name = video_key.replace('.mp4', '')

        for item in items:
            # Safely grab the basic fields
            query_id = item.get('query_id', '')
            question = item.get('question', '')
            ground_truth = item.get('ground_truth', '')
            
            # Grab nested annotation fields
            annotations = item.get('annotations', {})
            pipeline_mode = annotations.get('pipeline_mode', '')
            category = annotations.get('cognitive_category', '')
            
            # 3. Process timestamps & remove exact duplicates
            raw_segments = item.get('timestamp_segments', [])
            unique_timestamps = []
            seen = set()
            
            for seg in raw_segments:
                time_str = f"{seg.get('start', '')} - {seg.get('end', '')}"
                
                # Check for 100% match before adding
                if time_str not in seen:
                    seen.add(time_str)
                    unique_timestamps.append(time_str)
            
            # Join the unique timestamps into a single readable string
            # You can change ", " to "\n" if you want them on separate lines in Excel
            timestamp_str = " | ".join(unique_timestamps) 
            
            # 4. Append the cleaned data as a row dictionary
            rows.append({
                'query_id': query_id,
                'video_name': video_name,
                'pipeline_mode': pipeline_mode,
                'category': category,
                'timestamp': timestamp_str,
                'question': question,
                'ground_truth': ground_truth
            })

    # 5. Convert to a DataFrame and export to Excel
    df = pd.DataFrame(rows)
    df.to_excel(output_filepath, index=False, engine='openpyxl')
    print(f"Success! Data exported to {output_filepath}")

# Run the function
if __name__ == "__main__":
    # Change 'data.json' to whatever your actual json file is named
    convert_json_to_excel('/home/Pupil/dataset_curation/dataset/queries_db/final_1k/final_consolidated_1k.json', '/home/Pupil/dataset_curation/dataset/queries_db/final_1k/final_1k_dataset_queries.xlsx')