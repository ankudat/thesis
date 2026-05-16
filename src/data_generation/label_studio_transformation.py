import json
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Input: The raw data from the Gemini generator
INPUT_FILE = r"C:\thesis\data\processed\german_financial_data_cleaned.json"
# Output: The specific file to upload to Label Studio
OUTPUT_FILE = r"C:\thesis\data\processed\label_studio_final_import.json"

def create_label_studio_import():
    # 2. LOAD DATA
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    ls_tasks = []
    
    print(f"Loaded {len(raw_data)} documents. Converting to Label Studio format...")

    # 3. CONVERT ALL DOCUMENTS TO ANNOTATIONS
    for entry in raw_data:
        results = []
        
        # Safely get entities (defaults to an empty list if none exist)
        for ent in entry.get("entities", []):
            results.append({
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": ent["start"],
                    "end": ent["end"],
                    "text": ent["text"],
                    "labels": [ent["label"]]
                }
            })

        # Create task with the 'annotations' key
        ls_tasks.append({
            "data": {
                "text": entry.get("text", ""),
                "meta_temp": entry.get("meta_temp", "Unknown"),
                "raw_text": entry.get("raw_content", "")
            },
            "annotations": [{
                "result": results,
                "was_cancelled": False,
                "ground_truth": False
            }]
        })

    # 4. SAVE OUTPUT
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(ls_tasks, f, indent=2, ensure_ascii=False)

    print("-" * 30)
    print(f"Success!")
    print(f"Total tasks created: {len(ls_tasks)}")
    print(f"File saved to: {OUTPUT_FILE}")
    print("-" * 30)

if __name__ == "__main__":
    create_label_studio_import()