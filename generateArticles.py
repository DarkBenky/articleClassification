from addArticles import addToDataset
import ollama
import csv
import time
import random
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

USE_CLOUD = True
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

CLOUD_WORKERS = [
    {"model": "mistralai/mistral-small-3.2-24b-instruct", "provider": "deepinfra/fp8"},
    {"model": "qwen/qwen3.5-flash-02-23",                "provider": "alibaba"},
    {"model": "openai/gpt-oss-120b",                     "provider": "deepinfra/bf16"},
    {"model": "google/gemma-4-26b-a4b-it",               "provider": "deepinfra/fp8"},
    {"model": "mistralai/mistral-nemo",                  "provider": "deepinfra/fp8"},
]

CLOUD_DELAY = 0.0
CLOUD_MAX_RETRIES = 3

MODEL = "gemma4:e2b"

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
        # f"specific local place names, and at least one direct quote. "
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
        "PROXY WAR",
        "ALLIANCE",
        "TERRITORIAL DISPUTE",
        "ESPIONAGE",
        "FOREIGN POLICY",
        "COUP",
        "REGIME CHANGE",
        "DEBT CRISIS",
        "RECESSION",
        "TRADE WAR",
        "FISCAL POLICY",
        "MONETARY POLICY",
        "FOREIGN INVESTMENT",
        "STOCK MARKET",
        "PRIVATIZATION",
        "ECONOMIC SANCTIONS",
        "ARMS RACE",
        "MILITARY AID",
        "WAR CRIMES",
        "NAVAL CONFLICT",
        "AIR STRIKES",
        "INSURGENCY",
        "PEACEKEEPING",
        "DRONE WARFARE",
        "INTELLIGENCE",
        "DEMOGRAPHICS",
        "POPULATION",
        "ECONOMIC DATA",
        "CRIME STATISTICS",
        "MIGRATION TRENDS",
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

    if USE_CLOUD:
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
        from openai import OpenAI
        cloud_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        active = [w["model"] for w in CLOUD_WORKERS]
        print(f"Using {len(CLOUD_WORKERS)} cloud worker(s): {', '.join(active)}")
    else:
        print(f"Using local model: {MODEL} (ollama)")

    print("Running infinite loop — press Ctrl+C to stop.")
    print("-" * 60)

    lock  = threading.Lock()
    stats = {"counter": 0, "errors": 0, "start_time": time.time()}

    def run_worker(worker_cfg):
        model    = worker_cfg["model"] if USE_CLOUD else MODEL
        provider = worker_cfg.get("provider")
        while True:
            fips_code, location = random.choice(fips_items)
            subject = random.choice(SUBJECTS)
            type_   = random.choice(TYPES)
            style   = random.choice(STYLES)
            prompt  = createPrompt(subject, type_, style, location)
            article_text = None
            try:
                if USE_CLOUD:
                    wait = 5
                    for attempt in range(1, CLOUD_MAX_RETRIES + 1):
                        try:
                            resp = cloud_client.chat.completions.create(
                                model=model,
                                extra_body={
                                    "provider": {"order": [provider], "allow_fallbacks": False},
                                    "enable_thinking": False,
                                    "top_k": 20,
                                },
                                messages=[
                                    {"role": "system", "content": "You are a professional news writer. Never use emojis, emoticons, bullet points, hashtags, or special decorative characters in your output."},
                                    {"role": "user", "content": prompt},
                                ],
                                temperature=0.7,
                                top_p=0.8,
                                presence_penalty=1.5,
                            )
                            article_text = resp.choices[0].message.content.strip()
                            if CLOUD_DELAY > 0:
                                time.sleep(CLOUD_DELAY)
                            break
                        except Exception as api_err:
                            err_str = str(api_err)
                            if "per-day" in err_str or "per_day" in err_str:
                                print(f"\n[{model}] DAILY LIMIT HIT. Stopping this worker.")
                                return
                            if "429" in err_str and attempt < CLOUD_MAX_RETRIES:
                                print(f"  [{model}] 429 — retrying in {wait}s (attempt {attempt}/{CLOUD_MAX_RETRIES})")
                                time.sleep(wait)
                                wait = min(wait * 2, 120)
                            else:
                                raise
                else:
                    response = ollama.chat(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": "You are a professional news writer. Never use emojis, emoticons, bullet points, hashtags, or special decorative characters in your output."},
                            {"role": "user", "content": prompt},
                        ],
                        options={"temperature": 1.25, "top_p": 0.95, "top_k": 64},
                    )
                    article_text = response.message.content.strip()

                if article_text:
                    with lock:
                        addToDataset({"text": article_text, "location": fips_code, "category": subject, "model": model})
                        stats["counter"] += 1
                        cnt  = stats["counter"]
                        errs = stats["errors"]
                    elapsed = time.time() - stats["start_time"]
                    if cnt % 10 == 0:
                        apm = cnt / elapsed * 60 if elapsed > 0 else 0
                        print(f"[{cnt:,} generated] {apm:.1f} art/min | errors: {errs}")

            except Exception as e:
                with lock:
                    stats["errors"] += 1
                print(f"  [ERROR][{model}] {location} ({fips_code}): {e}")

    workers = CLOUD_WORKERS if USE_CLOUD else [{"model": MODEL, "provider": None}]
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = [executor.submit(run_worker, w) for w in workers]
        try:
            for f in futures:
                f.result()
        except KeyboardInterrupt:
            print("\nStopped.")
    


