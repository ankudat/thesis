import json
import re
import os
import time
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATION
# ==========================================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found! Please check your .env file.")

BASE_DIR = r"C:\thesis"
INPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "german_financial_data_raw.json")
FINAL_CLEANED_FILE = os.path.join(BASE_DIR, "data", "processed", "german_financial_data_cleaned.json")
REPORT_FILE = os.path.join(BASE_DIR, "results", "logs", "audit_report_detailed.txt")

# Ensure directories exist
os.makedirs(os.path.dirname(FINAL_CLEANED_FILE), exist_ok=True)
os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)

client = genai.Client(api_key=API_KEY)

# Stats Tracking
audit_stats = {
    "Deterministic": {"Low": 0, "Medium": 0, "High": 0, "Unknown": 0},
    "Semantic": {"Low": 0, "Medium": 0, "High": 0, "Unknown": 0},
    "Total_Input": {"Low": 0, "Medium": 0, "High": 0, "Unknown": 0}
}
deterministic_log = [] 
semantic_log = []

# ==========================================
# 2. STAGE 1: DETERMINISTIC RULES (Regex & Syntax)
# ==========================================
def rule_based_filter(data):
    tags = ["PER", "LOC", "ORG", "IBAN", "DATE", "EMAIL", "PHONE", "MONEY", "JOB", "AGE", "NATION", "EDU"]
    valid_markers = [f"<{t}>" for t in tags] + [f"</{t}>" for t in tags]
    
    clean_list = []
    print(f"Stage 1: Running Deterministic Checker on {len(data)} records...")

    for entry in data:
        temp = entry.get("meta_temp", "Unknown")
        audit_stats["Total_Input"][temp] = audit_stats["Total_Input"].get(temp, 0) + 1
        
        raw = entry.get("raw_content", "")
        entities = entry.get("entities", [])
        is_valid = True
        reasons = []

        # Check: Tag Balance
        for tag in tags:
            if raw.count(f"<{tag}>") != raw.count(f"</{tag}>"):
                is_valid = False
                reasons.append(f"Unbalanced <{tag}> tags")
        
        # Check: Hallucinated/Malformed Tags
        all_brackets = re.findall(r"<[^>]*>", raw)
        for b in all_brackets:
            if b not in valid_markers:
                is_valid = False
                reasons.append(f"Invalid tag: {b}")

        # Check: Bracket Leaks inside entities
        for ent in entities:
            txt = ent.get("text", "")
            if "<" in txt or ">" in txt:
                is_valid = False
                reasons.append(f"Bracket leak in entity: '{txt[:15]}...'")

        if is_valid:
            clean_list.append(entry)
        else:
            audit_stats["Deterministic"][temp] = audit_stats["Deterministic"].get(temp, 0) + 1
            deterministic_log.append({
                "id": entry.get("id", "N/A"),
                "temp": temp,
                "reason": "; ".join(reasons)
            })
            
    return clean_list

# ==========================================
# 3. STAGE 2: SEMANTIC AUDIT (LLM)
# ==========================================
def llm_semantic_audit(batch):
    """
    Identifies records with critical tagging or logic errors.
    Is lenient toward typos and realistic CRM shorthand.
    """

    audit_payload = [
        {
            "id": record.get("id"),
            "raw_content": record.get("raw_content")
        }
        for record in batch
    ]

    prompt = f"""
    You are a Data Quality Auditor for a Swiss Banking NER dataset.
    Review the 'raw_content' of these records and identify ONLY those that MUST be deleted.

    IMPORTANT: You must look specifically at the 'raw_content' field to evaluate the XML tags, as this is where the tagged entities are stored.

    ### DELETE IF:
    - Tagging Error: A currency (USD, CHF, EUR) is tagged as <LOC> instead of <MONEY>.
    - Tagging Error: A company/org is tagged as <PER>.
    - Logic Error: An IBAN is clearly fake (doesn't start with CH).
    - Logic Error: The content is not related to Swiss Corporate Banking or is in English.
    - Nesting: Tags are nested (e.g., <ORG><PER>...</PER></ORG>).

    ### KEEP IF (Acceptable Noise):
    - Typos: (e.g., 'Dringned' for 'Dringend') - these are realistic for hasty logs.
    - Dates: (e.g., '01.07.' without a year) - these are standard in CRM notes.
    - Style: Bullet points or fragments are allowed.

    DATA TO AUDIT:
    {json.dumps(audit_payload, indent=2, ensure_ascii=False)}

    RESPONSE FORMAT:
    Return ONLY a JSON list of objects.
    Example: [{{ "id": "doc_00001", "reason": "Currency CHF mislabeled as <LOC> in raw_content" }}]
    If no errors are found, return [].
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            config=types.GenerateContentConfig(
                temperature=0.0, 
                response_mime_type="application/json"
            ),
            contents=[prompt]
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"LLM Audit Error: {e}")
        return []

# ==========================================
# 4. MAIN EXECUTION & REPORTING
# ==========================================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # --- STAGE 1 ---
    filtered_data = rule_based_filter(raw_data)

    # --- STAGE 2 ---
    print(f"Stage 2: Running Semantic Audit on {len(filtered_data)} records...")
    final_kill_ids = []
    batch_size = 10
    
    for i in tqdm(range(0, len(filtered_data), batch_size)):
        batch = filtered_data[i : i + batch_size]
        failed_entries = llm_semantic_audit(batch)
        
        for fail in failed_entries:
            doc_id = fail.get("id")
            original_entry = next((x for x in batch if x["id"] == doc_id), None)
            if original_entry:
                temp = original_entry.get("meta_temp", "Unknown")
                audit_stats["Semantic"][temp] = audit_stats["Semantic"].get(temp, 0) + 1
                semantic_log.append({
                    "id": doc_id, "temp": temp, "reason": fail.get("reason", "Logic error")
                })
                final_kill_ids.append(doc_id)
        
        time.sleep(1.2) # Avoid Rate Limits

    # --- DATA SAVING ---
    final_data = [e for e in filtered_data if e["id"] not in final_kill_ids]
    with open(FINAL_CLEANED_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    # --- REPORT GENERATION ---
    with open(REPORT_FILE, "w", encoding="utf-8") as r:
        r.write("="*70 + "\n")
        r.write("DETAILED DATA AUDIT REPORT\n")
        r.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        r.write("="*70 + "\n\n")

        r.write("### 1. SUMMARY STATISTICS\n")
        r.write(f"{'Temp Level':<12} | {'Input':<8} | {'Rule Del':<10} | {'LLM Del':<10} | {'Final'}\n")
        r.write("-" * 60 + "\n")
        for lvl in ["Low", "Medium", "High"]:
            inp = audit_stats["Total_Input"].get(lvl, 0)
            rd = audit_stats["Deterministic"].get(lvl, 0)
            ld = audit_stats["Semantic"].get(lvl, 0)
            r.write(f"{lvl:<12} | {inp:<8} | {rd:<10} | {ld:<10} | {inp-rd-ld}\n")

        r.write("\n### 2. [SECTION A] DETERMINISTIC DELETIONS (Syntax/Regex)\n")
        r.write(f"{'ID':<10} | {'Temp':<8} | {'Reason'}\n")
        r.write("-" * 80 + "\n")
        for log in deterministic_log:
            r.write(f"{log['id']:<10} | {log['temp']:<8} | {log['reason']}\n")

        r.write("\n### 3. [SECTION B] SEMANTIC DELETIONS (LLM Logic)\n")
        r.write(f"{'ID':<10} | {'Temp':<8} | {'Reason'}\n")
        r.write("-" * 80 + "\n")
        for log in semantic_log:
            r.write(f"{log['id']:<10} | {log['temp']:<8} | {log['reason']}\n")

    print(f"\n SUCCESS!")
    print(f"Final Clean Records: {len(final_data)} (Total Deleted: {len(deterministic_log) + len(semantic_log)})")
    print(f"Report location: {REPORT_FILE}")

if __name__ == "__main__":
    main()