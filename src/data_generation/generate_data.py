import os
import json
import re
import time
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ==========================================
# 1. CONFIGURATION
# ==========================================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Set GEMINI_API_KEY in the environment or in a .env file.")

# DATASET SETTINGS
TOTAL_SAMPLES_NEEDED = 3000
BATCH_SIZE = 10  
MODEL_NAME = "gemini-2.5-pro" 

# PATH SETUP (Windows Format)
OUTPUT_DIR = r"C:\thesis\data\raw"
# Changed back to standard .json as requested
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "german_financial_data_raw.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Client
client = genai.Client(api_key=API_KEY)

# ==========================================
# 2. DEFINING TEMPERATURE LEVELS
# ==========================================
TEMP_SETTINGS = [
    {
        "level": "Low", 
        "temp": 0.30, 
        "desc": "Formal Visit Report. Full sentences, perfect Standard German. Objective tone. Official records."
    },
    {
        "level": "Medium", 
        "temp": 0.70, 
        "desc": "Standard CRM Note in German. Concise, professional banking terminology. Uses industry abbreviations."
    },
    {
        "level": "High", 
        "temp": 0.85, 
        "desc": "Hasty Quick-Log in German. Bullet points, fragments. INTENTIONALLY include realistic typos (e.g., 'habne', 'fianziell') to simulate hasty typing. Keep XML tags perfect."
    }
]

# ==========================================
# 3. PROMPTS (UPDATED FOR HIGHER QUALITY)
# ==========================================
SYSTEM_INSTRUCTION_BASE = """
You are a Swiss Corporate & Institutional Clients (CIC) Relationship Manager.
Task: Generate **Client Visit Reports** and **Internal CRM Notes** regarding your portfolio of Swiss corporate clients.

**LANGUAGE MANDATE:**
1. ALL CONTENT MUST BE WRITTEN IN STANDARD GERMAN (Hochdeutsch).
2. Do NOT generate notes in English.
3. You may use Swiss banking terminology (e.g., "Traktanden", "Pendenzen").

**### CRITICAL ANNOTATION RULES (STRICT XML SYNTAX) ###**
1. **Flat Structure Only:** NEVER nest tags. 
    - WRONG: <ORG><PER>Hans</PER> AG</ORG> 
    - RIGHT: <PER>Hans</PER> von der <ORG>UBS AG</ORG>
2. **No Attributes:** NEVER add attributes. 
    - WRONG: <MONEY currency="CHF"> 
    - RIGHT: <MONEY>CHF 50'000</MONEY>
3. **Closing Tags & No Hallucinations:** Every opening tag MUST have a matching closing tag immediately after the entity. NEVER invent new tags (e.g., DO NOT use <GEO>). Use ONLY the supported tags listed below.
4. **Standalone Currency Codes are NOT Entities:** Do NOT tag standalone currency codes (like "USD", "EUR", "CHF") without an amount. They are NOT money, NOT locations, and NOT nationalities — leave them completely untagged. Only tag monetary expressions that include a numeric value.
    - WRONG: Importe in <MONEY>USD</MONEY>
    - WRONG: Geschäfte in <LOC>CHF</LOC>
    - WRONG: <NATION>EUR</NATION>-Raum
    - RIGHT: Importe in USD (no tag)
    - RIGHT: <MONEY>CHF 50'000</MONEY>
    - RIGHT: <MONEY>USD 1.2 Mio.</MONEY>
5. **No Titles or Salutations in PER tags:** NEVER include salutations (Herr, Frau) or academic/professional titles (Dr., Prof., CEO) inside the <PER> tag.
    - WRONG: <PER>Herr Dr. Beat Weber</PER>
    - RIGHT: Herr Dr. <PER>Beat Weber</PER>
6. **Exclude Articles:** NEVER include definite or indefinite articles (der, die, das, ein, eine) inside the tags.
    - WRONG: <ORG>der UBS AG</ORG>
    - RIGHT: der <ORG>UBS AG</ORG>
7. **Exclude Surrounding Punctuation:** NEVER include commas, colons, or end-of-sentence periods inside the tag, UNLESS the period is strictly part of an abbreviation (e.g., "Mio.", "GmbH.", "Inc.").
    - WRONG: in <LOC>Zürich,</LOC> und <LOC>Bern.</LOC>
    - RIGHT: in <LOC>Zürich</LOC>, und <LOC>Bern</LOC>.
8. **No Sub-Word Tagging:** Do not split hyphenated words with tags. If an entity is part of a hyphenated compound, tag the entire compound based on its primary meaning.
    - WRONG: <EDU>ETH</EDU>-Ingenieur
    - RIGHT: <JOB>ETH-Ingenieur</JOB>
9. **Full Company Names:** ALWAYS include legal entity suffixes (AG, GmbH, SA, Ltd.) inside the <ORG> tag.
    - WRONG: <ORG>Alpen Tech</ORG> AG
    - RIGHT: <ORG>Alpen Tech AG</ORG>

**Supported Tags:**
- <PER>: Person names (e.g., Hans Müller). Strictly exclude titles/salutations.
- <EMAIL>: Email addresses.
- <PHONE>: Phone numbers.
- <IBAN>: IBANs (Must start with CH).
- <MONEY>: Monetary values AND standalone currency codes (e.g., CHF, USD, EUR, 50k).
- <JOB>: Job titles.
- <AGE>: Ages/Birth years.
- <NATION>: Nationalities.
- <EDU>: Education.
- <LOC>: Addresses, cities, cantons, countries (Do NOT tag currencies here).
- <ORG>: Company names.
- <DATE>: Dates.

**### FORMATTING RULES ###**
1. NO HEADERS or SUBJECT LINES (e.g., No "Note 1", No "Betreff:").
2. START DIRECTLY with the text content.
3. SEPARATOR: Use "###SEPARATOR###" strictly between notes.

**### GOLD STANDARD EXAMPLES (FOLLOW THIS FORMAT STRICTLY) ###**

[Example 1 - Formal Style]
Am <DATE>12.03.2024</DATE> traf ich Herrn Dr. <PER>Beat Weber</PER>, den <JOB>CFO</JOB> der <ORG>Alpen Tech AG</ORG>, in <LOC>Zürich</LOC>. Wir besprachen die Erhöhung der Kreditlimite auf <MONEY>CHF 2.5 Mio.</MONEY> (Gegenwert in <MONEY>USD</MONEY>). Er bestätigte, dass die <JOB>ETH-Absolventin</JOB> Frau <PER>Sarah Müller</PER> neue <JOB>CEO</JOB> wird. Bitte <EMAIL>s.mueller@alpentech.ch</EMAIL> für KYC kontaktieren.

[Example 2 - Hasty Style]
Tel mit <PER>Rolf</PER> (<PHONE>079 555 22 11</PHONE>). Hat Stress wegen der <ORG>Baugruppe Nord</ORG>. Will <MONEY>50k</MONEY> oder <MONEY>EUR</MONEY> sofort auf <IBAN>CH93 0070 0111 2222 3333 4</IBAN> überweisen. <LOC>Bern</LOC> macht Druck. Ist <AGE>60-jährig</AGE> und wirkt müde.

**### END EXAMPLES ###**

STYLE INSTRUCTION:
{style_desc}
"""

USER_PROMPT_TEMPLATE = """
Generate {n} distinct **CIC Client Notes** in GERMAN, separating them ONLY with "###SEPARATOR###".

**OUTPUT LANGUAGE: GERMAN**

INSTRUCTIONS:
1. Mix Direct Identifiers (<PER>, <EMAIL>, <PHONE>, <IBAN>) and Indirect Identifiers (<JOB>, <MONEY>, <ORG>) heavily in every note.
2. Randomly select scenarios from this list:

SCENARIOS TO COVER:
- **Treasury/Cash:** FX, Cash Management, Deposits, Festgeld, Akkreditiv.
- **Financing:** TEF (Trade & Export Finance), Lending, Leasing, Hypotheken, Kreditlimiten.
- **Corporate:** Nachfolge, Spin-offs, Governance Meetings, Wechsel Geschäftsführung.
- **Interaction:** Betriebsbesichtigung, Geschäftsmodell Firma, Umsatz, EBITDA, momentaner Zustand.
- **Lifecycle/Ops:** Kauf Liegenschaften, Kontosaldierung, Kontoeröffnung, zusätzliche Konten, Einreichung Jahresabschlüsse.
- **Admin:** Kartenlimite Erhöhung, Fehlende Dokumente, Unterschriften (rechtsverbindlich), E-Banking-Vertrag, Zugriffsberechtigungen.

Begin generation now:
"""

# ==========================================
# 4. PARSING LOGIC 
# ==========================================
def parse_tagged_text(raw_text, doc_id_start, temp_level):
    if not raw_text: return []
    
    raw_messages = raw_text.split("###SEPARATOR###")
    full_records = []
    current_id = doc_id_start
    
    tag_pattern = re.compile(r"<\s*(PER|LOC|ORG|IBAN|DATE|EMAIL|PHONE|MONEY|JOB|AGE|NATION|EDU)\s*>(.*?)<\/\s*\1\s*>", re.DOTALL)
    
    for msg in raw_messages:
        msg = msg.strip()
        msg = re.sub(r"^(?:\**)?Note\s+\d+\**", "", msg, flags=re.IGNORECASE).strip()
        msg = re.sub(r"^Betreff:.*?\n", "", msg, flags=re.IGNORECASE).strip()
        msg = msg.lstrip("*# \n\t")

        if not msg or not re.search(r"<\s*[A-Z]+\s*>", msg):
            continue
        
        entities = []
        clean_text = ""
        last_pos = 0
        
        for match in tag_pattern.finditer(msg):
            tag_name = match.group(1)
            # Strip leading/trailing spaces from the entity text
            content = match.group(2).strip() 
            start_tag_start, end_tag_end = match.span()
            
            clean_text += msg[last_pos:start_tag_start]
            entity_start = len(clean_text)
            
            clean_text += content
            entity_end = len(clean_text)
            
            entities.append({
                "start": entity_start,
                "end": entity_end,
                "label": tag_name,
                "text": content
            })
            last_pos = end_tag_end
            
        clean_text += msg[last_pos:]
        
        full_records.append({
            "id": f"doc_{current_id:05d}", # 5-digit padding for 3000+ samples
            "meta_temp": temp_level, 
            "text": clean_text.strip(),
            "entities": entities,
            "raw_content": msg.strip()
        })
        current_id += 1
        
    return full_records

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    print(f"--- Starting Generation of {TOTAL_SAMPLES_NEEDED} Samples ---")
    
    # Store all parsed objects in memory
    all_parsed_objects = []
    doc_counter = 1
    samples_per_level = TOTAL_SAMPLES_NEEDED // len(TEMP_SETTINGS)

    for setting in TEMP_SETTINGS:
        level, temp, desc = setting["level"], setting["temp"], setting["desc"]
        print(f"\nProcessing Style: {level} (T={temp})")
        
        current_sys = SYSTEM_INSTRUCTION_BASE.format(style_desc=desc)
        num_batches = (samples_per_level + BATCH_SIZE - 1) // BATCH_SIZE
        
        for _ in tqdm(range(num_batches), desc=f"Progress {level}"):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME, 
                    config=types.GenerateContentConfig(
                        system_instruction=current_sys,
                        temperature=temp,
                        max_output_tokens=5000 
                    ),
                    contents=[USER_PROMPT_TEMPLATE.format(n=BATCH_SIZE)]
                )
                
                if response.text:
                    recs = parse_tagged_text(response.text, doc_counter, level)
                    # Extend the main list with the newly parsed records
                    all_parsed_objects.extend(recs)
                    doc_counter += len(recs)
                
                # Sleep to respect the 150 RPM rate limit
                time.sleep(1.5)
                
            except Exception as e:
                print(f"\nError encountered during batch: {e}")
                time.sleep(10)

    # Write the complete list to a standard JSON file at the very end
    print(f"\nSaving {len(all_parsed_objects)} samples to file...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_parsed_objects, f, indent=2, ensure_ascii=False)

    print(f"Success! All data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()