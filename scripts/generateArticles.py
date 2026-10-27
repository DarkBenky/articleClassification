"""Generate synthetic news articles and append them to the corpus."""

import random
import re
import time
from collections import defaultdict

import requests

from addArticles import addToDataset
from categoryLabels import validateCategories
from groupLocations import flipCodeToName
from project_config import VALIDATE_PATH

CHAT_URL = "http://localhost:8013/v1/chat/completions"
MODELS_URL = "http://localhost:8013/v1/models"
MIN_WORD_COUNT = 50
GENERATE_VALIDATION_DATASET = True

try:
    from langdetect import detect as _lang_detect
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


def getModel() -> str:
    response = requests.get(MODELS_URL, timeout=10)
    response.raise_for_status()
    data = response.json()
    models = data.get("data") or data.get("models", [])
    if not models:
        raise ValueError("No model is available at localhost:8013.")
    return models[0].get("id") or models[0].get("model") or models[0]["name"]


def createPrompt(subject, type_, style, location, date):
    use_indirect = random.random() < 0.35
    if use_indirect:
        location_instruction = (
            f"set in {location}, but do NOT explicitly name the country or region. "
            f"Instead, convey the location only through indirect cues: "
            f"use the capital city, well-known landmarks, local institutions, "
            f"currency, prominent politicians, or colloquial names that strongly imply {location}. "
            f"A reader familiar with the region should recognise it, but the country name must not appear."
        )
    else:
        location_instruction = (
            f"set specifically in {location}. "
            f"Ground every detail firmly in {location} — do not make it generic or global."
        )
    prompt = (
        f"Write a {style.lower()}, {type_.lower()} news article about {subject.lower()} "
        f"{location_instruction} "
        f"The article must read like a real published piece from {date}: include a headline, "
        f"dateline with the city/country, named local officials or organisations, "
        f"Length: 200-400 words. Output only the article text, no commentary. "
        f"Avoid using any emojis or special characters. The article should be in English."
    )
    return prompt, use_indirect


def callModel(model: str, prompt: str) -> str:
    response = requests.post(
        CHAT_URL,
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional news writer. Never use emojis, emoticons, "
                        "bullet points, hashtags, or special decorative characters in your output."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 1.25,
            "top_p": 0.95,
            "max_tokens": 8000,
            "reasoning_effort": "low",
        },
        timeout=360,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"].get("content")
    return content.strip() if content else ""


def clean_article(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def is_english(text):
    if not _LANGDETECT_AVAILABLE:
        return True
    try:
        return _lang_detect(text[:600]) == "en"
    except Exception:
        return True


def leaks_location(text, location):
    return bool(re.search(re.escape(location), text, re.IGNORECASE))


if __name__ == "__main__":
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

    validateCategories(SUBJECTS, "generator subjects")

    model = getModel()
    model_tag = model.rsplit("/", 1)[-1]
    fips_items = list(flipCodeToName.items())

    print(f"Using local model: {model}")
    print("Running single-threaded — press Ctrl+C to stop.")
    print("-" * 60)

    stats = {"counter": 0, "errors": 0, "start_time": time.time(),
             "indirect_total": 0, "indirect_leaks": 0}
    fips_counts = defaultdict(int)
    pair_counts = defaultdict(int)

    while True:
        try:
            fips_weights = [1.0 / (fips_counts.get(code, 0) + 1) for code, _ in fips_items]
            fips_code, location = random.choices(fips_items, weights=fips_weights, k=1)[0]

            subject_weights = [1.0 / (pair_counts.get((fips_code, s), 0) + 1) for s in SUBJECTS]
            subject = random.choices(SUBJECTS, weights=subject_weights, k=1)[0]

            type_ = random.choice(TYPES)
            style = random.choice(STYLES)
            year = random.randint(2020, 2026) if random.random() < 0.65 else random.randint(2000, 2035)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            date_str = f"{day}. {time.strftime('%B', time.strptime(str(month), '%m'))} {year}"
            prompt, use_indirect = createPrompt(subject, type_, style, location, date_str)

            article_text = clean_article(callModel(model, prompt))

            if not article_text:
                stats["errors"] += 1
                print(f"  [SKIP] {location}: empty response")
                continue

            if len(article_text.split()) < MIN_WORD_COUNT:
                stats["errors"] += 1
                print(f"  [SKIP] {location}: too short ({len(article_text.split())} words)")
                continue

            if not is_english(article_text):
                stats["errors"] += 1
                print(f"  [SKIP] {location}: non-English output")
                continue

            if use_indirect:
                stats["indirect_total"] += 1
                if leaks_location(article_text, location):
                    stats["indirect_leaks"] += 1

            if GENERATE_VALIDATION_DATASET:
                addToDataset({
                                "text": article_text,
                                "location": fips_code,
                                "category": subject,
                                "model": model_tag,
                            }, VALIDATE_PATH)
            else:
                addToDataset({
                    "text": article_text,
                    "location": fips_code,
                    "category": subject,
                    "model": model_tag,
                })
            fips_counts[fips_code] += 1
            pair_counts[(fips_code, subject)] += 1
            stats["counter"] += 1

            if stats["counter"] % 10 == 0:
                elapsed = time.time() - stats["start_time"]
                apm = stats["counter"] / elapsed * 60 if elapsed > 0 else 0
                ind_total = stats["indirect_total"]
                ind_success = (1 - stats["indirect_leaks"] / ind_total) * 100 if ind_total else 0
                print(f"[{stats['counter']:,} generated] {apm:.1f} art/min | "
                      f"errors: {stats['errors']} | indirect: {ind_success:.1f}% success")

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as exc:
            stats["errors"] += 1
            print(f"  [ERROR] {exc}")
