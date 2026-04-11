from addArticles import addToDataset
import ollama
import csv
import time
import random

# MODEL = "gemma4:e4b"
MODEL = "gemma4:e2b" # fast
# MODEL = "qwen3.5:2b" # peaty slow
# MODEL = "smollm2:360m" # really fast, but lower quality

def getCountries():
    flipCodeToName = {}
    NameToFlipCode = {}
    with open("fipsCodes.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["FIPS 10-4"].strip()
            name = row["Name"].strip()
            if code == "-" or not code:
                continue
            flipCodeToName[code] = name
            NameToFlipCode[name] = code
    return flipCodeToName, NameToFlipCode

def createPrompt(subject, type_, style, location):
    return (
        f"Write a {style.lower()}, {type_.lower()} news article about {subject.lower()} "
        f"set specifically in {location}. "
        f"The article must read like a real published piece: include a headline, "
        f"dateline with the city/country, named local officials or organisations, "
        f"specific local place names, and at least one direct quote. "
        f"Ground every detail firmly in {location} — do not make it generic or global. "
        f"Length: 200-400 words. Output only the article text, no commentary. "
        f"Avoid using any emojis or special characters. The article should be in English."
    )

if __name__ == "__main__":
    flipCodeToName, NameToFlipCode = getCountries()

    SUBJECTS = [
        "ECONOMY",
        "POLITICS",
        "ENVIRONMENT",
        "HEALTH",
        "TECHNOLOGY",
        "SPORTS",
        "CULTURE",
        "SCIENCE",
        "WORLD",
        "MILITARY",
        "DISASTER",
        "DIPLOMACY",
        "CRIME",
        "EDUCATION",
        "TRADE",
        "ENERGY",
        "TRANSPORT",
        "AGRICULTURE",
        "GEOPOLITICS",
        "HUMAN RIGHTS",
        "INFLATION",
        "UNEMPLOYMENT",
        "ELECTIONS",
        "CLIMATE CHANGE",
        "PUBLIC HEALTH",
        "CYBERSECURITY",
        "TERRORISM",
        "SANCTIONS",
        "NUCLEAR",
        "PROTESTS",
        "CORRUPTION",
        "BORDER SECURITY",
        "IMMIGRATION",
        "REFUGEES",
        "RELIGION",
        "POVERTY",
        "HOUSING",
        "BANKING",
        "OIL & GAS",
        "SUPPLY CHAIN",
        "CURRENCY",
        "ARTIFICIAL INTELLIGENCE",
        "SPACE",
        "NUCLEAR ENERGY",
        "MEDIA",
        "JUSTICE",
        "WATER RESOURCES",
        "INFRASTRUCTURE",
        "SOCIAL MEDIA",
        # Geopolitics / Diplomacy
        "PROXY WAR",
        "ALLIANCE",
        "TERRITORIAL DISPUTE",
        "ESPIONAGE",
        "FOREIGN POLICY",
        "COUP",
        "REGIME CHANGE",
        # Economics / Finance
        "DEBT CRISIS",
        "RECESSION",
        "TRADE WAR",
        "FISCAL POLICY",
        "MONETARY POLICY",
        "FOREIGN INVESTMENT",
        "STOCK MARKET",
        "PRIVATIZATION",
        "ECONOMIC SANCTIONS",
        # Military
        "ARMS RACE",
        "MILITARY AID",
        "WAR CRIMES",
        "NAVAL CONFLICT",
        "AIR STRIKES",
        "INSURGENCY",
        "PEACEKEEPING",
        "DRONE WARFARE",
        "INTELLIGENCE",
        # Statistics / Data-driven
        "DEMOGRAPHICS",
        "POPULATION",
        "ECONOMIC DATA",
        "CRIME STATISTICS",
        "MIGRATION TRENDS",
        # Other
        "FAMINE",
        "PANDEMIC",
        "SEPARATISM",
        "CIVIL WAR",
        "ORGANIZED CRIME",
        "MONEY LAUNDERING",
        "POLITICAL PRISONERS",
    ]

    TYPES = [
        "BREAKING NEWS",
        "ANALYSIS",
        "OPINION",
        "INTERVIEW",
        "REPORT",
        "EDITORIAL",
        "FEATURE",
        "CASE STUDY",
        "NEWSLETTER",
        "META-ANALYSIS",
        "REVIEW",
        "PREDICTION",
    ]

    STYLES = [
        "FORMAL",
        "INFORMAL",
        "CONCISE",
        "DETAILED",
        "NARRATIVE",
        "ANALYTICAL",
        "PERSUASIVE",
        "DESCRIPTIVE",
        "EXPLANATORY",
        "CRITICAL",
        "HUMOROUS",
        "SARCASTIC",
        "EMOTIONAL",
    ]


    fips_items = list(flipCodeToName.items())
    print("Running infinite loop — press Ctrl+C to stop.")
    print("-" * 60)
    counter = 0
    errors  = 0
    start_time = time.time()

    while True:
        fips_code, location = random.choice(fips_items)
        subject = random.choice(SUBJECTS)
        type_   = random.choice(TYPES)
        style   = random.choice(STYLES)
        prompt  = createPrompt(subject, type_, style, location)
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional news writer."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 1.25, "top_p": 0.95, "top_k": 64},
            )
            article_text = response.message.content.strip()
            addToDataset({
                "text": article_text,
                "location": fips_code,
                "category": subject,
            })
        except Exception as e:
            errors += 1
            print(f"  [ERROR] {location} ({fips_code}): {e}")
        counter += 1
        if counter % 10 == 0:
            elapsed = time.time() - start_time
            apm     = counter / elapsed * 60 if elapsed > 0 else 0
            print(
                f"[{counter:,} generated] "
                f"{apm:.1f} art/min | "
                f"errors: {errors}"
            )
    


