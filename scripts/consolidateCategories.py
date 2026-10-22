"""Map corpus category labels to CAMEO root codes."""

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

from project_config import CAMEO_CODES_PATH, PROCESSED_DATA_DIR, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "inception/mercury-decide"
NONE_KEY = "none"
MIN_COUNT = 125
WORKERS = 8  # parallel requests; keep low enough to avoid rate limits
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


TOPIC_HINTS = {
    "210": "Politics: government, elections, policy, diplomacy",
    "211": "Economy: business, finance, markets, commodities such as gold and oil, trade, earnings",
    "212": "Crime and law: crime, courts, justice, police",
    "213": "Sport: sports, matches, teams, athletes",
    "214": "Arts and culture: entertainment, music, film, media, celebrities",
    "215": "Science and tech: technology, science, AI, space, internet",
    "216": "Health: medicine, health, wellness, pandemic",
    "217": "Environment: climate, nature, water, energy",
    "218": "Society: general news, world, local, community, immigration",
    "219": "Conflict and war: military, war, terrorism, security",
    "220": "Education: schools, universities, learning",
    "221": "Lifestyle: travel, fashion, beauty, food, leisure",
    "222": "Weather: forecasts, storms, weather events",
    "223": "Religion: faith, churches, religious affairs",
    "224": "Labour: jobs, employment, unions, wages",
    "225": "Disaster: accidents, disasters, emergencies",
    "226": "Human interest: people, personal stories, curiosities",
    "227": "Other: anything not covered above",
}


def loadRoots() -> dict:
    roots, subs = {}, {}
    in_topics = False
    in_subjects = False
    with open(CAMEO_CODES_PATH, "r", encoding="utf-8") as file:
        for row in csv.reader(file):
            if len(row) != 2:
                continue
            code, name = row[0].strip(), row[1].strip()
            if not code:
                continue
            if len(code) == 2:
                in_topics = code == "21"
                in_subjects = code == "30"
                if in_topics or in_subjects:
                    continue
                roots[code] = name
            elif in_topics:
                roots.setdefault(code, name)
            elif in_subjects:
                roots[code] = name
            else:
                subs.setdefault(code[:2], []).append(name)
    return {
        code: TOPIC_HINTS.get(code) or f"{name}: " + ", ".join(subs.get(code, []))
        for code, name in roots.items()
    }


ROOTS = loadRoots()


def mapToCameo(category: str, minProb: float = 0.5) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

    criteria = dict(ROOTS)
    criteria[NONE_KEY] = "Not a usable category label (a URL, junk value, or empty text)"
    body = {
        "model": MODEL,
        "state": [f"News category label: {category}"],
        "questions": {
            "root": {
                "type": "choice",
                "instructions": "Which category fits this news category label?",
                "criteria": criteria,
            }
        },
    }
    r = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body),
        timeout=60,
    )
    if not r.ok:
        raise requests.HTTPError(f"{r.status_code} {r.reason}: {r.text[:300]}")

    answer = r.json()["answers"]["root"]
    choice = answer["choice"]
    if choice not in ROOTS or answer["probabilities"].get(choice, 0) < minProb:
        return NONE_KEY
    return choice


if __name__ == "__main__":
    SAVE_PATH = PROCESSED_DATA_DIR / "category_to_cameo.json"

    mapped_categories = {}
    if SAVE_PATH.exists():
        with open(SAVE_PATH, "r", encoding="utf-8") as file:
            mapped_categories = json.load(file)
        # drop cached codes that no longer exist in CODES.csv
        mapped_categories = {
            k: v for k, v in mapped_categories.items()
            if v in ROOTS or v == NONE_KEY
        }

    with open(PROCESSED_DATA_DIR / "unique_categories.json", "r", encoding="utf-8") as file:
        raw_categories = json.load(file)

    pending = [
        category
        for category, count in raw_categories.items()
        if category not in mapped_categories and count >= MIN_COUNT
    ]
    print(f"{len(pending)} categories to map with {WORKERS} workers.")

    i = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(mapToCameo, category): category for category in pending}
        for future in as_completed(futures):
            category = futures[future]
            try:
                mapped_categories[category] = future.result()
            except Exception as exc:
                print(f"[error] {category}: {exc}")
                continue

            i += 1
            if i % 10 == 0:
                print(f"Mapped {i}/{len(pending)} new categories ({len(mapped_categories)} total).")
                with open(SAVE_PATH, "w", encoding="utf-8") as file:
                    json.dump(mapped_categories, file, indent=4)

    with open(SAVE_PATH, "w", encoding="utf-8") as file:
        json.dump(mapped_categories, file, indent=4)
    print(f"Done. Mapped {i} new categories, {len(mapped_categories)} total.")
