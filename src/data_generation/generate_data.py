import os
import json
import re
import time
import random
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
    raise ValueError("API Key not found! Please check your .env file.")

# DATASET SETTINGS
TOTAL_SAMPLES_NEEDED = 3000
BATCH_SIZE = 10
MODEL_NAME = "gemini-2.5-pro"

# PATH SETUP (Windows Format)
OUTPUT_DIR = r"C:\thesis\data\raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "german_financial_data_raw.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Client
client = genai.Client(api_key=API_KEY)

# ==========================================
# 2. NAME & ENTITY POOLS (KEY IMPROVEMENT)
# ==========================================
# Large pools to inject into prompts per-batch, preventing repetition.

FIRST_NAMES_MALE = [
    "Daniel", "Thomas", "Michael", "Andreas", "Christian", "Martin", "Markus", "Peter",
    "Stefan", "Patrick", "Marco", "David", "Pascal", "Marcel", "Urs", "Marc",
    "Roger", "Bruno", "Roland", "Simon", "Beat", "Hans", "Christoph", "René",
    "Manuel", "Adrian", "Reto", "José", "Nicolas", "André", "Matthias", "Stephan",
    "Philipp", "Antonio", "Philippe", "Rolf", "Fabian", "Lukas", "Alexander", "Mario",
    "Michel", "Roman", "Samuel", "Florian", "Robert", "Olivier", "Oliver", "Benjamin",
    "Jürg", "Luca", "Tobias", "Dominik", "Sandro", "Kevin", "Walter", "Giuseppe",
    "Alain", "Claudio", "Christophe", "Carlos", "Fabio", "Alexandre", "Jean", "Stéphane",
    "Jan", "Heinz", "Paul", "Kurt", "Laurent", "Pierre", "Roberto", "Yves",
    "Francesco", "Werner", "Raphael", "Eric", "Andrea", "Ivan", "Julien", "Rudolf",
    "Frédéric", "Sébastien", "Josef", "Remo", "Alessandro", "Cédric", "Bernhard", "Thierry",
    "Felix", "Vincent", "Sven", "Jonas", "Sebastian", "Richard", "Anton", "Ulrich",
    "Ali", "Jonathan", "Giovanni", "Patrik", "Mathias", "Stefano", "Sascha", "Paulo",
    "François", "Jörg", "Alfred", "Dominique", "Dario", "Frank", "Claude", "Nicola",
    "Pedro", "Erich", "Franz", "Michele", "Jérôme", "Luis", "Guido", "Davide",
]

FIRST_NAMES_FEMALE = [
    "Maria", "Sandra", "Claudia", "Andrea", "Nicole", "Monika", "Daniela", "Barbara",
    "Karin", "Christine", "Manuela", "Silvia", "Anna", "Susanne", "Brigitte", "Ursula",
    "Sarah", "Cornelia", "Gabriela", "Anita", "Franziska", "Nathalie", "Corinne", "Ana",
    "Patricia", "Martina", "Laura", "Sabrina", "Sonja", "Isabelle", "Esther", "Marianne",
    "Alexandra", "Beatrice", "Fabienne", "Yvonne", "Ruth", "Elisabeth", "Doris", "Melanie",
    "Nadine", "Sabine", "Jacqueline", "Caroline", "Rita", "Petra", "Tanja", "Irene",
    "Katharina", "Angela", "Sara", "Marie", "Regula", "Simone", "Stefanie", "Nadia",
    "Verena", "Catherine", "Jessica", "Carmen", "Tamara", "Erika", "Anne", "Vanessa",
    "Eva", "Marina", "Julia", "Denise", "Heidi", "Bettina", "Christina", "Céline",
    "Jasmin", "Chantal", "Elena", "Rahel", "Diana", "Eveline", "Judith", "Sophie",
    "Valérie", "Stephanie", "Mirjam", "Jennifer", "Anja", "Nadja", "Cristina", "Stéphanie",
    "Janine", "Patrizia", "Priska", "Michèle", "Astrid", "Ramona", "Sonia", "Véronique",
    "Monica", "Nina", "Edith", "Susanna", "Sibylle", "Rosa", "Katja", "Maja",
    "Sylvie", "Renate", "Marion", "Miriam", "Dominique", "Isabel", "Carla", "Pia",
    "Margrit", "Michelle", "Iris", "Rosmarie", "Myriam", "Michaela", "Linda", "Aline",
]

LAST_NAMES = [
    "Müller", "Meier", "Schmid", "Keller", "Weber", "Schneider", "Huber", "Meyer",
    "Steiner", "da Silva", "Fischer", "Gerber", "Baumann", "Brunner", "Frei", "Zimmermann",
    "Moser", "Graf", "Widmer", "Wyss", "Ferreira", "Roth", "Pereira", "Bucher",
    "Baumgartner", "Bachmann", "Suter", "Kaufmann", "Studer", "Berger", "Lüthi", "Bühler",
    "Kunz", "Krasniqi", "Lehmann", "Hofer", "Marti", "dos Santos", "Berisha", "Rodrigues",
    "Arnold", "Koch", "Christen", "Frey", "Wüthrich", "Egli", "Gashi", "Zürcher",
    "Fuchs", "Pfister", "Gasser", "Fernandes", "Stalder", "Koller", "Schweizer", "Martin",
    "Peter", "Bieri", "Gomes", "Maurer", "Kohler", "Wenger", "Furrer", "Burri",
    "Vogel", "Michel", "Leuenberger", "Rüegg", "Martins", "Schär", "Egger", "Garcia",
    "Hunziker", "Lopes", "Schuler", "Kälin", "Ammann", "Hofmann", "Hess", "Hug",
    "Tanner", "Gisler", "Sutter", "Favre", "Wagner", "Blaser", "Hauser", "Oliveira",
    "Alves", "Ribeiro", "Schmidt", "Silva", "Hartmann", "Gonçalves", "Shala", "Senn",
    "Flückiger", "Lang", "Stucki", "Odermatt", "Pinto", "Bajrami", "Siegenthaler", "Fankhauser",
    "Teixeira", "Scherrer", "Zbinden", "Morina", "Marques", "Ramadani", "Sommer", "Zaugg",
    "Imhof", "Portmann", "Küng", "da Costa", "Santos", "Rodriguez", "Ackermann", "Nguyen",
    "Schärer", "Scheidegger", "Vogt", "Schwarz", "Jost", "Schenk", "Rey", "Liechti",
    "Kuhn", "Schumacher", "Hasler", "Hofstetter", "Costa", "Giger", "Weiss", "Staub",
    "Seiler", "Stocker", "Röthlisberger", "Betschart", "Herzog", "Schnyder", "Lüscher", "Fässler",
    "Wittwer", "Wolf", "Marty", "Haas", "Zehnder", "Stadelmann", "Dias", "Fernandez",
    "Stöckli", "Schwab", "Käser", "Schaller", "Bühlmann", "Martinez", "Gonzalez", "Weibel",
    "Näf", "Kaiser", "Häfliger", "Steiger", "Rohner", "Ulrich", "Bernasconi", "Rossi",
    "Gloor", "Stutz", "Bosshard", "Stettler", "Lutz", "Rohrer", "Walker", "Beck",
    "Lanz", "Grob", "Mäder", "Tobler", "Steffen", "Blum", "Brügger", "Aeschlimann",
    "Sigrist", "Meister", "Osmani", "Jenni", "Ziegler", "Eichenberger", "de Oliveira", "Lopez",
    "Kuster", "Sieber", "Ademi", "Kessler", "Siegrist", "Wicki", "Shabani", "Bolliger",
]

COMPANY_PREFIXES = [
    "Alpen", "Alpine", "Aqua", "Astra", "Berg", "Bio", "Blau", "Brücken",
    "Central", "Chrono", "Clar", "Delta", "Diamant", "Digi", "Eco", "Edel",
    "Elektro", "Elite", "Euro", "First", "Flora", "Forst", "Gastro", "Global",
    "Granit", "Grün", "Heli", "Helvetia", "Horizon", "Hydro", "Inno", "Inter",
    "Jura", "Klima", "Kraft", "Kristall", "Lago", "Linth", "Matterhorn", "Medico",
    "Metro", "Micro", "Monta", "Navi", "Neo", "Nexus", "Nova", "Omega", "Optima",
    "Peak", "Pharma", "Pionier", "Planet", "Pola", "Präzis", "Prima", "Pro",
    "Quarz", "Rapid", "Reno", "Rhein", "Riviera", "Robo", "Roto", "Saline",
    "Saphir", "Saturn", "Schild", "Senn", "Signal", "Solar", "Spektrum", "Stahl",
    "Stern", "Stratos", "Swiss", "Techno", "Terra", "Thermo", "Titan", "Topaz",
    "Trans", "Trio", "Turbo", "Ultra", "Urban", "Vecto", "Ventus", "Vero",
    "Viso", "Volta", "Weiss", "Wetter", "Xenon", "Zenit", "Zentral", "Züri",
]

COMPANY_SUFFIXES = [
    "Bau", "Consult", "Design", "Dynamics", "Electronics", "Engineering",
    "Export", "Finance", "Food", "Freight", "Handel", "Holding", "Import",
    "Industries", "Innovation", "Invest", "IT", "Klinik", "Logistik",
    "Maschinenbau", "Mechanik", "Media", "Metall", "Mobilität", "Partners",
    "Pharma", "Precision", "Produktion", "Robotics", "Services", "Software",
    "Solutions", "Sport", "Systems", "Tech", "Textil", "Trade", "Transport",
    "Treuhand", "Ventures", "Werkzeuge",
]

LEGAL_FORMS = ["AG", "GmbH", "SA", "Sàrl", "& Co. KG", "& Cie."]

SWISS_CITIES = [
    "Zürich", "Genf", "Basel", "Bern", "Lausanne", "Winterthur", "Luzern",
    "St. Gallen", "Lugano", "Biel/Bienne", "Thun", "Köniz", "La Chaux-de-Fonds",
    "Schaffhausen", "Freiburg", "Chur", "Neuchâtel", "Vernier", "Uster", "Sion",
    "Emmen", "Kriens", "Rapperswil-Jona", "Zug", "Dübendorf", "Dietikon",
    "Frauenfeld", "Wil", "Aarau", "Baden", "Olten", "Solothurn", "Grenchen",
    "Langenthal", "Burgdorf", "Bellinzona", "Locarno", "Martigny", "Montreux",
    "Nyon", "Morges", "Vevey", "Yverdon-les-Bains", "Delémont", "Brig-Glis",
    "Wädenswil", "Horgen", "Thalwil", "Kloten", "Opfikon", "Wallisellen",
    "Arth", "Rotkreuz", "Cham", "Baar", "Stans", "Sarnen", "Altdorf",
    "Schwyz", "Glarus", "Appenzell", "Herisau", "Gossau", "Buchs",
    "Davos", "Interlaken", "Grindelwald", "Meiringen", "Spiez", "Brienz",
]

JOB_TITLES = [
    "CEO", "CFO", "COO", "CTO", "CIO", "CHRO", "CLO", "CMO", "CSO",
    "Geschäftsführer", "Geschäftsführerin",
    "Verwaltungsratspräsident", "Verwaltungsratspräsidentin",
    "Verwaltungsrat", "Verwaltungsrätin",
    "Finanzchef", "Finanzchefin",
    "Leiter Finanzen", "Leiterin Finanzen",
    "Leiter Treasury", "Leiterin Treasury",
    "Leiter Buchhaltung", "Leiterin Buchhaltung",
    "Leiter Export", "Leiterin Export",
    "Leiter Einkauf", "Leiterin Einkauf",
    "Leiter IT", "Leiterin IT",
    "Head of Treasury", "Head of Finance", "Head of Operations",
    "Head of Compliance", "Head of Legal", "Head of Sales",
    "Inhaber", "Inhaberin",
    "Gründer", "Gründerin",
    "Mitgründer", "Mitgründerin",
    "Teilhaber", "Teilhaberin",
    "Projektleiter", "Projektleiterin",
    "Prokurist", "Prokuristin",
    "Buchhalter", "Buchhalterin",
    "Treasurer", "Controller", "Controllerin",
    "Export Manager", "Export Managerin",
    "Office Manager", "Office Managerin",
    "Betriebsleiter", "Betriebsleiterin",
    "Produktionsleiter", "Produktionsleiterin",
    "Personalchef", "Personalchefin",
    "Syndikus", "Syndika",
    "General Counsel", "VP Finance", "VP Operations",
    "Managing Director", "Director",
]

EDUCATION_LABELS = [
    "HSG", "ETH", "EPFL", "Universität Zürich", "Universität Bern",
    "Universität Basel", "Universität Genf", "Universität Lausanne",
    "Universität St. Gallen", "Universität Freiburg", "Universität Luzern",
    "Universität Neuenburg", "USI Lugano", "ZHAW", "FHNW", "HWZ",
    "BFH", "HSLU", "OST", "SUPSI", "HES-SO",
    "MBA", "EMBA", "CAS", "MAS", "DAS",
    "INSEAD", "IMD", "London Business School",
    "HSG-Absolvent", "HSG-Absolventin", "ETH-Ingenieur", "ETH-Ingenieurin",
    "ETH-Abschluss", "ETH-Diplom", "EPFL-Abschluss",
    "lic. oec.", "lic. iur.", "Dr. oec.", "Dr. iur.", "dipl. Ing. ETH",
]

NATIONALITIES = [
    "schweizerischer", "schweizerische", "Schweizer",
    "deutscher", "deutsche", "Deutscher", "Deutsche",
    "österreichischer", "österreichische",
    "französischer", "französische",
    "italienischer", "italienische",
    "britischer", "britische",
    "amerikanischer", "amerikanische",
    "brasilianischer", "brasilianische",
    "indischer", "indische",
    "chinesischer", "chinesische",
    "japanischer", "japanische",
    "koreanischer", "koreanische",
    "türkischer", "türkische",
    "portugiesischer", "portugiesische",
    "spanischer", "spanische",
    "polnischer", "polnische",
    "kroatischer", "kroatische",
    "serbischer", "serbische",
    "niederländischer", "niederländische",
    "schwedischer", "schwedische",
    "dänischer", "dänische",
]


def generate_random_names(n=8):
    """Pick n random full names from the pools, ensuring no duplicates per batch."""
    names = []
    used = set()
    all_first = FIRST_NAMES_MALE + FIRST_NAMES_FEMALE
    for _ in range(n):
        while True:
            first = random.choice(all_first)
            last = random.choice(LAST_NAMES)
            full = f"{first} {last}"
            if full not in used:
                used.add(full)
                names.append(full)
                break
    return names


def generate_random_companies(n=5):
    """Generate n unique random company names."""
    companies = []
    used = set()
    for _ in range(n):
        while True:
            prefix = random.choice(COMPANY_PREFIXES)
            suffix = random.choice(COMPANY_SUFFIXES)
            legal = random.choice(LEGAL_FORMS[:3])  # Bias toward AG/GmbH/SA
            name = f"{prefix}{suffix} {legal}"
            if name not in used:
                used.add(name)
                companies.append(name)
                break
    return companies


def generate_random_cities(n=5):
    """Pick n random cities."""
    return random.sample(SWISS_CITIES, min(n, len(SWISS_CITIES)))


def generate_random_jobs(n=5):
    """Pick n random job titles."""
    return random.sample(JOB_TITLES, min(n, len(JOB_TITLES)))


# ==========================================
# 3. TEMPERATURE LEVELS
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
# 4. PROMPTS (IMPROVED FOR DIVERSITY)
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
3. **Closing Tags & No Hallucinations:** Every opening tag MUST have a matching closing tag immediately after the entity. NEVER invent new tags. Use ONLY the supported tags listed below.
4. **Currency vs. Money Distinction:**
    - Standalone currency codes (EUR, USD, CHF, GBP, JPY, etc.) used as REFERENCES to a currency (e.g., "ein Konto in USD") should NOT be tagged at all.
    - ONLY tag <MONEY> when there is an ACTUAL monetary amount: <MONEY>CHF 50'000</MONEY>, <MONEY>USD 1.2 Mio.</MONEY>, <MONEY>50k</MONEY>.
    - WRONG: ein Konto in <MONEY>USD</MONEY>   →   RIGHT: ein Konto in USD
    - WRONG: den <MONEY>EUR</MONEY>/<MONEY>CHF</MONEY>-Kurs   →   RIGHT: den EUR/CHF-Kurs
    - RIGHT: eine Überweisung von <MONEY>EUR 250'000</MONEY>
5. **No Titles or Salutations in PER tags:** NEVER include salutations (Herr, Frau) or academic/professional titles (Dr., Prof., CEO) inside the <PER> tag.
    - WRONG: <PER>Herr Dr. Beat Weber</PER>
    - RIGHT: Herr Dr. <PER>Beat Weber</PER>
6. **Exclude Articles:** NEVER include definite or indefinite articles (der, die, das, ein, eine) inside the tags.
7. **Exclude Surrounding Punctuation:** NEVER include commas, colons, or end-of-sentence periods inside the tag, UNLESS the period is strictly part of an abbreviation (e.g., "Mio.", "GmbH.", "Inc.").
8. **No Sub-Word Tagging:** Do not split hyphenated words with tags. If an entity is part of a hyphenated compound, tag the entire compound.
    - WRONG: <EDU>ETH</EDU>-Ingenieur → RIGHT: <JOB>ETH-Ingenieur</JOB>
9. **Full Company Names:** ALWAYS include legal entity suffixes (AG, GmbH, SA, Ltd.) inside the <ORG> tag.
    - WRONG: <ORG>Alpen Tech</ORG> AG → RIGHT: <ORG>Alpen Tech AG</ORG>
10. **LOC vs. Countries:** Use <LOC> for geographic locations (cities, cantons, countries, regions, continents). Countries and regions are LOC.
11. **AGE consistency:** Always tag the full age expression including unit words.
    - WRONG: <AGE>65</AGE>-jährig → RIGHT: <AGE>65-jährig</AGE>
    - RIGHT: <AGE>65 Jahre</AGE> alt
12. **Consistent ORG naming:** Each company should have ONE canonical name within a single note. Do not alternate between "AG" and "GmbH" for the same company.

**Supported Tags:**
- <PER>: Person names (e.g., Hans Müller). Strictly exclude titles/salutations.
- <EMAIL>: Email addresses.
- <PHONE>: Phone numbers.
- <IBAN>: IBANs (Must start with CH, exactly 21 characters).
- <MONEY>: Monetary amounts with values (NOT standalone currency codes).
- <JOB>: Job titles (including compound forms like ETH-Ingenieur).
- <AGE>: Full age expressions (e.g., "65-jährig", "65 Jahre").
- <NATION>: Nationality adjectives (e.g., "deutscher", "französische").
- <EDU>: Education institutions and qualifications (e.g., "HSG", "MBA", "ETH-Abschluss").
- <LOC>: Cities, cantons, countries, regions, continents.
- <ORG>: Company names including legal suffix.
- <DATE>: Dates.

**### FORMATTING RULES ###**
1. NO HEADERS or SUBJECT LINES (e.g., No "Note 1", No "Betreff:").
2. START DIRECTLY with the text content.
3. SEPARATOR: Use "###SEPARATOR###" strictly between notes.

**### EXAMPLE (FOLLOW THIS FORMAT STRICTLY) ###**

Am <DATE>12.03.2024</DATE> traf ich Herrn Dr. <PER>Beat Weber</PER>, den <JOB>CFO</JOB> der <ORG>Alpen Tech AG</ORG>, in <LOC>Zürich</LOC>. Wir besprachen die Erhöhung der Kreditlimite auf <MONEY>CHF 2.5 Mio.</MONEY>. Die <JOB>ETH-Absolventin</JOB> Frau <PER>Sarah Müller</PER> wird neue <JOB>CEO</JOB>. Bitte <EMAIL>s.mueller@alpentech.ch</EMAIL> für KYC kontaktieren.

**### END EXAMPLE ###**

STYLE INSTRUCTION:
{style_desc}
"""

# The user prompt now injects random entities to force diversity
USER_PROMPT_TEMPLATE = """
Generate {n} distinct **CIC Client Notes** in GERMAN, separating them ONLY with "###SEPARATOR###".

**MANDATORY: USE THESE SPECIFIC NAMES, COMPANIES, AND LOCATIONS in your notes (distribute them across the {n} notes):**

**Person Names to use:** {names}
**Company Names to use:** {companies}
**Cities to use:** {cities}
**Job Titles to use:** {jobs}

You may also invent ADDITIONAL names/companies beyond these, but you MUST use the ones listed above. 
Do NOT reuse the same person name across multiple notes — each note should feature different people.

SCENARIOS TO COVER (randomly mix):
- Treasury/Cash: FX, Cash Management, Deposits, Festgeld, Akkreditiv
- Financing: TEF, Lending, Leasing, Hypotheken, Kreditlimiten
- Corporate: Nachfolge, Spin-offs, Governance, Wechsel Geschäftsführung
- Interaction: Betriebsbesichtigung, Geschäftsmodell, Umsatz, EBITDA
- Lifecycle/Ops: Kauf Liegenschaften, Kontosaldierung, Kontoeröffnung, Jahresabschlüsse
- Admin: Kartenlimite, Fehlende Dokumente, Unterschriften, E-Banking, Zugriffsberechtigungen

Begin generation now:
"""

# ==========================================
# 5. PARSING LOGIC (IMPROVED)
# ==========================================
def parse_tagged_text(raw_text, doc_id_start, temp_level):
    if not raw_text:
        return []

    raw_messages = raw_text.split("###SEPARATOR###")
    full_records = []
    current_id = doc_id_start

    tag_pattern = re.compile(
        r"<\s*(PER|LOC|ORG|IBAN|DATE|EMAIL|PHONE|MONEY|JOB|AGE|NATION|EDU)\s*>"
        r"(.*?)"
        r"<\/\s*\1\s*>",
        re.DOTALL
    )

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
            "id": f"doc_{current_id:05d}",
            "meta_temp": temp_level,
            "text": clean_text.strip(),
            "entities": entities,
            "raw_content": msg.strip()
        })
        current_id += 1

    return full_records


# ==========================================
# 6. POST-PROCESSING VALIDATION
# ==========================================
def validate_record(record):
    """Flag common issues for optional review."""
    issues = []
    text = record["text"]

    for ent in record["entities"]:
        # Check offset alignment
        extracted = text[ent["start"]:ent["end"]]
        if extracted != ent["text"]:
            issues.append(f"OFFSET MISMATCH: '{ent['text']}' vs '{extracted}'")

        # Flag standalone currency codes tagged as MONEY
        if ent["label"] == "MONEY" and ent["text"] in ("EUR", "CHF", "USD", "GBP", "JPY", "CNY", "CAD", "AUD"):
            issues.append(f"STANDALONE_CURRENCY_AS_MONEY: '{ent['text']}'")

        # Flag Herr/Frau in PER
        if ent["label"] == "PER" and (ent["text"].startswith("Herr ") or ent["text"].startswith("Frau ")):
            issues.append(f"TITLE_IN_PER: '{ent['text']}'")

    return issues


# ==========================================
# 7. MAIN EXECUTION
# ==========================================
def main():
    print(f"--- Starting Generation of {TOTAL_SAMPLES_NEEDED} Samples ---")

    all_parsed_objects = []
    doc_counter = 1
    samples_per_level = TOTAL_SAMPLES_NEEDED // len(TEMP_SETTINGS)
    validation_log = []

    for setting in TEMP_SETTINGS:
        level, temp, desc = setting["level"], setting["temp"], setting["desc"]
        print(f"\nProcessing Style: {level} (T={temp})")

        current_sys = SYSTEM_INSTRUCTION_BASE.format(style_desc=desc)
        num_batches = (samples_per_level + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_num in tqdm(range(num_batches), desc=f"Progress {level}"):
            try:
                # Generate FRESH random entities for EVERY batch
                batch_names = generate_random_names(n=8)
                batch_companies = generate_random_companies(n=5)
                batch_cities = generate_random_cities(n=5)
                batch_jobs = generate_random_jobs(n=5)

                prompt = USER_PROMPT_TEMPLATE.format(
                    n=BATCH_SIZE,
                    names=", ".join(batch_names),
                    companies=", ".join(batch_companies),
                    cities=", ".join(batch_cities),
                    jobs=", ".join(batch_jobs),
                )

                response = client.models.generate_content(
                    model=MODEL_NAME,
                    config=types.GenerateContentConfig(
                        system_instruction=current_sys,
                        temperature=temp,
                        max_output_tokens=5000
                    ),
                    contents=[prompt]
                )

                if response.text:
                    recs = parse_tagged_text(response.text, doc_counter, level)

                    # Validate each record
                    for rec in recs:
                        issues = validate_record(rec)
                        if issues:
                            validation_log.append({
                                "id": rec["id"],
                                "issues": issues
                            })

                    all_parsed_objects.extend(recs)
                    doc_counter += len(recs)

                time.sleep(1.5)

            except Exception as e:
                print(f"\nError in batch {batch_num}: {e}")
                time.sleep(10)

    # Save main dataset
    print(f"\nSaving {len(all_parsed_objects)} samples to file...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_parsed_objects, f, indent=2, ensure_ascii=False)

    # Save validation log
    log_file = os.path.join(OUTPUT_DIR, "validation_issues.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(validation_log, f, indent=2, ensure_ascii=False)

    print(f"Success! Data saved to: {OUTPUT_FILE}")
    print(f"Validation log ({len(validation_log)} records with issues) saved to: {log_file}")

    # Print summary stats
    from collections import Counter
    label_counts = Counter()
    for rec in all_parsed_objects:
        for ent in rec["entities"]:
            label_counts[ent["label"]] += 1
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
