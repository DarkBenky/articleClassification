import ast
import atexit
import hashlib
import html
import json
import os
import pickle
import random
import re
import sys
import unicodedata
from itertools import islice

import requests
from dotenv import load_dotenv

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")),
)
from project_config import (
    ARTICLE_HASHES_PATH, CLEAN_CORPUS_PATH, CORPUS_PATH, REPO_ROOT,
)
from categoryLabels import loadCameoCodes

load_dotenv(REPO_ROOT / ".env")

CAME_CODES, CAMEO_NAMES = loadCameoCodes()
NAME_BY_CODE = dict(zip(CAME_CODES, CAMEO_NAMES))

def chunkCodes(codes, maxSize):
    count = (len(codes) + maxSize - 1) // maxSize
    size = (len(codes) + count - 1) // count
    return [codes[i:i + size] for i in range(0, len(codes), size)]

LEAF_CODES = [
    code for code in CAME_CODES
    if len(code) >= 3 and not any(other != code and other.startswith(code) for other in CAME_CODES)
]
CATEGORY_CHUNKS = chunkCodes(LEAF_CODES, 255)

CHUNK_SIZE = 8196
MIN_WORD_COUNT = 40
MIN_QUALITY = 0.5
MIN_LABEL_CONFIDENCE = 0.7
MIN_CATEGORY_CONFIDENCE = 0.7
JUDGE_URL = "https://openrouter.ai/api/alpha/decisions"
JUDGE_MODEL = "baseten/mammoth-1:free"
JUDGE_MAX_CHARS = 8192

def readChunks(chunkSize):
    with open(CORPUS_PATH, encoding="utf-8") as f:
        while True:
            lines = list(islice(f, chunkSize))
            if not lines:
                return
            records = [ast.literal_eval(line) for line in lines]
            random.shuffle(records)
            yield records

def removeSpecialCharacters(text):
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\ufeff]", "", text)

def removeHtmlTags(text):
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)

def removeRedundantWhitespace(text):
    return re.sub(r"\s+", " ", text).strip()

def judgeText(text, label):
    apiKey = os.environ.get("OPENROUTER_API_KEY", "")
    if not apiKey:
        raise ValueError("OPENROUTER_API_KEY is not set")
    body = {
        "model": JUDGE_MODEL,
        "state": text[:JUDGE_MAX_CHARS],
        "questions": {
            "quality": {
                "type": "noul",
                "instructions": "Is this high quality news article text?",
                "criteria": {
                    "true": "Well-formed reporting prose about a real event",
                    "false": "Navigation, boilerplate, paywall or cookie notice, listing, truncated stub, or incoherent text",
                },
            },
            "label_correct": {
                "type": "noul",
                "instructions": f"Does the article support the assigned category label '{label}'?",
                "criteria": {
                    "true": f"The article is primarily about {label}",
                    "false": "The article is about something else, or the label is a placeholder",
                },
            },
        },
    }
    for index, chunk in enumerate(CATEGORY_CHUNKS):
        body["questions"][f"category_{index}"] = {
            "type": "choice",
            "instructions": "Which category best describes this article?",
            "criteria": {code: NAME_BY_CODE[code] for code in chunk},
        }
    response = requests.post(
        JUDGE_URL,
        headers={
            "Authorization": f"Bearer {apiKey}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=60,
    )
    response.raise_for_status()
    answers = response.json()["answers"]
    chosen, chosenProb = "", 0.0
    for index in range(len(CATEGORY_CHUNKS)):
        answer = answers[f"category_{index}"]
        prob = answer["probabilities"].get(answer["choice"], 0.0)
        if prob > chosenProb:
            chosen, chosenProb = answer["choice"], prob
    return {
        "quality": answers["quality"]["noul"],
        "labelCorrect": answers["label_correct"]["noul"],
        "category": chosen,
        "categoryProb": chosenProb,
    }

# remove html tags, remove special characters, remove redundant whitespace
# skip records under MIN_WORD_COUNT words or already hashed
# ask decision model once:
# q = is this text high quality news article? [noul]
# x = is this category correct? [noul] prob
# cat = select correct category [all specific CODES.csv categories, generic parents excluded]
# if q < MIN_QUALITY:
#    skip
# if x < MIN_LABEL_CONFIDENCE and cat prob >= MIN_CATEGORY_CONFIDENCE:
#    use cat, keep old label as labelOriginal
# save to cleaned corpus
# save hash of this article to avoid duplicates
seenHashes = None

def loadHashes():
    global seenHashes
    if seenHashes is None:
        if ARTICLE_HASHES_PATH.exists():
            with open(ARTICLE_HASHES_PATH, "rb") as f:
                seenHashes = pickle.load(f)
        else:
            seenHashes = set()
    return seenHashes

def saveHashes():
    if seenHashes is not None:
        ARTICLE_HASHES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ARTICLE_HASHES_PATH, "wb") as f:
            pickle.dump(seenHashes, f)

atexit.register(saveHashes)

def processChunk(chunk):
    hashes = loadHashes()
    kept = 0
    CLEAN_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLEAN_CORPUS_PATH, "a", encoding="utf-8") as out:
        for record in chunk:
            text = record.get("text")
            if not isinstance(text, str):
                continue
            text = removeRedundantWhitespace(
                removeSpecialCharacters(removeHtmlTags(text))
            )
            if len(text.split()) < MIN_WORD_COUNT:
                continue
            digest = hashlib.sha256(text.lower().encode("utf-8")).hexdigest()
            if digest in hashes:
                continue
            verdict = judgeText(text, record.get("label", ""))
            if verdict["quality"] < MIN_QUALITY:
                continue
            hashes.add(digest)
            cleaned = dict(record)
            cleaned["text"] = text
            if verdict["labelCorrect"] < MIN_LABEL_CONFIDENCE and verdict["categoryProb"] >= MIN_CATEGORY_CONFIDENCE:
                cleaned["labelOriginal"] = cleaned.get("label")
                cleaned["label"] = verdict["category"]
            out.write(repr(cleaned) + "\n")
            kept += 1
    return kept