import json
import os
import re

# ================= CONFIGURATION =================
VIDEO_DIR = os.getcwd()
MANIFEST_FILES = [
    'dataset_manifest1.json', 
    'dataset_manifest2.json', 
    'dataset_manifest3.json', 
    'dataset_manifest4.json'
]
OUTPUT_JSON = 'final_dataset.json'

# ================= CATEGORIZATION RULES =================
# Defined order for Indexing: Region -> Source
def match_regex(filename, regex):
    return bool(re.search(regex, filename))

CATEGORIES = [
    # --- ASIA / INDIA ---
    { "region": "Asia / India", "source": "NPTEL Engineering Graphics", "matcher": lambda f: match_regex(f, r'^w\d+_l\d+') },
    { "region": "Asia / India", "source": "NPTEL Metallurgy", "matcher": lambda f: "mod_01_lec_" in f and any(x in f for x in ["phase", "crystal", "defect", "structure"]) },
    { "region": "Asia / India", "source": "NPTEL Analog Circuits", "matcher": lambda f: match_regex(f, r'^lecture_\d+') and "elec2141" not in f },
    { "region": "Asia / India", "source": "NPTEL Robotics", "matcher": lambda f: match_regex(f, r'mod\d+lec\d+') },
    { "region": "Asia / India", "source": "IITB Organic Chemistry", "matcher": lambda f: any(x in f for x in ["organic", "mechanism", "chem"]) and "rust" not in f and "nitrogen" not in f },

    # --- OCEANIA / AUSTRALIA ---
    { "region": "Oceania / Australia", "source": "Eddie Woo", "matcher": lambda f: any(x in f for x in ["prisms", "3d_coordinate", "relating_vectors", "vector_arithmetic"]) },
    { "region": "Oceania / Australia", "source": "ELEC2141 Digital Circuits", "matcher": lambda f: "elec2141" in f },
    { "region": "Oceania / Australia", "source": "UNSW Calculus", "matcher": lambda f: match_regex(f, r'^ch\d+_pr\d+') },

    # --- NORTH AMERICA ---
    { "region": "North America", "source": "SmarterEveryday", "matcher": lambda f: "smarter_every_day" in f },
    { "region": "North America", "source": "Veritasium", "matcher": lambda f: any(x in f for x in ["chaos", "perplexing", "knot_theory", "butterfly"]) },
    { "region": "North America", "source": "Real Engineering", "matcher": lambda f: "the_insane_engineering" in f },

    # --- EUROPE / SWISS ---
    { "region": "Europe / Swiss", "source": "ETH Zürich", "matcher": lambda f: "eth_zürich" in f or match_regex(f, r'^week_\d+_lecture') },
    { "region": "Europe / Swiss", "source": "Post Apocalyptic Inventor", "matcher": lambda f: any(x in f for x in ["rust", "raft", "liquid_nitrogen", "sci_fi", "led", "experiments"]) },

    # --- AFRICA / OTHERS ---
    { "region": "Africa / Uganda", "source": "FOG Accountancy", "matcher": lambda f: any(x in f for x in ["account", "cashbook", "double_entry", "partner"]) },
    { "region": "Africa / Stanford", "source": "Tadashi Tokieda (Topology)", "matcher": lambda f: "topology" in f },
    { "region": "Africa / Geo", "source": "Geo-Strategy", "matcher": lambda f: "civilization" in f }
]

def sanitize_filename(filename):
    """Removes '001_' prefix if present to restore original name."""
    match = re.match(r'^\d{3}_(.+)', filename)
    if match:
        return match.group(1)
    return filename

def main():
    print("--- 🏁 Starting Final Dataset Generation ---")
    
    # 1. RESTORE ORIGINAL FILENAMES (If they were changed)
    # We must ensure the file on disk matches the original name in JSON
    disk_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    restored_count = 0
    
    for f in disk_files:
        clean_name = sanitize_filename(f)
        if f != clean_name:
            try:
                os.rename(os.path.join(VIDEO_DIR, f), os.path.join(VIDEO_DIR, clean_name))
                restored_count += 1
            except OSError as e:
                print(f"❌ Error restoring {f}: {e}")
    
    if restored_count > 0:
        print(f"✅ Restored {restored_count} files to their original names.")
    else:
        print("✅ Files appear to be in their original state (no prefixes).")

    # 2. LOAD & MERGE MANIFESTS
    all_metadata = []
    seen_filenames = set()
    
    for manifest in MANIFEST_FILES:
        if os.path.exists(manifest):
            try:
                with open(manifest, 'r') as f:
                    data = json.load(f)
                    for entry in data:
                        # Clean the filename in the manifest just in case
                        entry['filename'] = sanitize_filename(entry['filename'])
                        
                        if entry['filename'] not in seen_filenames:
                            all_metadata.append(entry)
                            seen_filenames.add(entry['filename'])
            except json.JSONDecodeError:
                print(f"⚠️  Warning: Could not decode {manifest}")
    
    print(f"📊 Loaded {len(all_metadata)} unique videos from manifests.")

    # 3. CATEGORIZE & INDEX
    final_list = []
    pool = all_metadata.copy()
    current_index = 1
    
    # Process strictly in the order of CATEGORIES to maintain your grouping
    for cat in CATEGORIES:
        # Find matches
        matches = [v for v in pool if cat['matcher'](v['filename'])]
        
        # Sort alphabetically to ensure deterministic indexing
        matches.sort(key=lambda x: x['filename'])
        
        for video in matches:
            # Assign Index and Category info
            video['index'] = current_index
            video['region'] = cat['region']
            video['source_category'] = cat['source']
            
            final_list.append(video)
            current_index += 1
            
            # Remove from pool
            pool.remove(video)
            
    # 4. HANDLE UNCATEGORIZED (Add them at the end)
    if pool:
        print(f"ℹ️  {len(pool)} videos fell into 'Uncategorized'. Appending them at the end.")
        pool.sort(key=lambda x: x['filename'])
        for video in pool:
            video['index'] = current_index
            video['region'] = "Uncategorized"
            video['source_category'] = "Unknown"
            final_list.append(video)
            current_index += 1

    # 5. WRITE FINAL JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_list, f, indent=4)
        
    print(f"\n✅ SUCCESS! Generated '{OUTPUT_JSON}' with {len(final_list)} entries.")
    print("Files on disk have been preserved with their original names.")

if __name__ == "__main__":
    main()