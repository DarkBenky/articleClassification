from addArticles import addToDataset
import ollama
import csv
import time

# MODEL = "gemma4:e4b"
# MODEL = "gemma4:e2b" // fast
# MODEL = "qwen3.5:2b" # peaty slow
MODEL = "smollm2:360m"

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


    totalCount = len(flipCodeToName) * len(SUBJECTS) * len(TYPES) * len(STYLES) * 3
    print(f"Total articles to generate: {totalCount:,}")
    print("-" * 60)
    counter = 0
    errors  = 0
    start_time = time.time()

    for fips_code, location in flipCodeToName.items().__reversed__():
        for subject in SUBJECTS:
            for type_ in TYPES:
                for style in STYLES:
                    for _ in range(3):  # generate 3 articles per combination
                        prompt = createPrompt(subject, type_, style, location)
                        try:
                            response = ollama.chat(
                                model=MODEL,
                                messages=[
                                    {"role": "system", "content": "You are a professional news writer."},
                                    {"role": "user", "content": prompt},
                                ],
                                options={"temperature": 1.15, "top_p": 0.95, "top_k": 64},
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
                            elapsed  = time.time() - start_time
                            apm      = counter / elapsed * 60 if elapsed > 0 else 0
                            eta_s    = (totalCount - counter) / (counter / elapsed) if counter > 0 else 0
                            eta_h    = int(eta_s // 3600)
                            eta_m    = int((eta_s % 3600) // 60)
                            pct      = counter / totalCount * 100
                            print(
                                f"[{counter:>7,}/{totalCount:,} ({pct:.1f}%)] "
                                f"{apm:.1f} art/min | "
                                f"ETA: {eta_h}h {eta_m:02d}m | "
                                f"errors: {errors}"
                            )

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Done. {counter:,} attempts | {counter - errors:,} saved | {errors} errors | {total_time/60:.1f} min total")
    


