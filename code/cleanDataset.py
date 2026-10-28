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
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
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

RESUME_PATH = "./resume.txt"
FAILED_PATH = "./failed.txt"
BATCH_SIZE = 8196
MAX_PARALLEL = 8

load_dotenv(REPO_ROOT / ".env")

CAME_CODES, CAMEO_NAMES = loadCameoCodes()
NAME_BY_CODE = dict(zip(CAME_CODES, CAMEO_NAMES))

def shortName(name):
    name = re.split(r"\s*[\(\-\u2013]\s*", name)[0].strip(" ,.")
    return " ".join(name.split()[:3])

SHORT_NAME_BY_CODE = {code: shortName(name) for code, name in NAME_BY_CODE.items()}

def chunkCodes(codes, maxSize):
    if not codes:
        return []
    count = (len(codes) + maxSize - 1) // maxSize
    size = (len(codes) + count - 1) // count
    return [codes[i:i + size] for i in range(0, len(codes), size)]

LEAF_CODES = [
    code for code in CAME_CODES
    if len(code) >= 3 and not any(other != code and other.startswith(code) for other in CAME_CODES)
]
TOPIC_CODES = [code for code in CAME_CODES if len(code) == 3 and 210 <= int(code) <= 227]

# category question: "topic" = 18 GENERAL NEWS TOPICS, "leaf" = all leaf codes, "off" = skip
CATEGORY_MODE = "topic"
CATEGORY_CODES = {"topic": TOPIC_CODES, "leaf": LEAF_CODES, "off": []}[CATEGORY_MODE]

MIN_WORD_COUNT = 40
MIN_QUALITY = 0.5
MIN_LABEL_CONFIDENCE = 0.7
MIN_CATEGORY_CONFIDENCE = 0.7
JUDGE_URL = "https://openrouter.ai/api/alpha/decisions"
JUDGE_MODELS = [
    ("perplexity/pplx-decider-v1.2", 255),
    ("cloudflare/clef-flash", 26),
]
JUDGE_MAX_CHARS = 8192
RETRY_STATUS = (429, 500, 502, 503, 504)
MAX_RETRIES = 6
judgeModelIndex = 0
judgeModelLock = threading.Lock()
totalCost = 0.0
costLock = threading.Lock()
failedCalls = 0
failedLock = threading.Lock()

def readChunks(chunkSize, skipLines = 0):
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for _ in range(skipLines):
            if not f.readline():
                return
        index = skipLines
        while True:
            lines = list(islice(f, chunkSize))
            if not lines:
                return
            records = [(index + i, ast.literal_eval(line)) for i, line in enumerate(lines)]
            index += len(lines)
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
    questions = {
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
    }
    global judgeModelIndex, totalCost
    while True:
        with judgeModelLock:
            model, maxOptions = JUDGE_MODELS[judgeModelIndex]
        chunks = chunkCodes(CATEGORY_CODES, maxOptions)
        body = {"model": model, "state": text[:JUDGE_MAX_CHARS], "questions": dict(questions)}
        for index, chunk in enumerate(chunks):
            body["questions"][f"category_{index}"] = {
                "type": "choice",
                "instructions": "Which category best describes this article?",
                "criteria": {code: SHORT_NAME_BY_CODE[code] for code in chunk},
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
        reason = response.text.lower()
        if response.status_code == 404 or (
            response.status_code == 400
            and any(word in reason for word in ("does not exist", "only accepts", "at most"))
        ):
            with judgeModelLock:
                if judgeModelIndex >= len(JUDGE_MODELS) - 1:
                    raise RuntimeError(f"no judge model available: {response.text[:200]}")
                judgeModelIndex += 1
            continue
        response.raise_for_status()
        break
    payload = response.json()
    answers = payload["answers"]
    with costLock:
        totalCost += payload.get("usage", {}).get("cost", 0.0)
    chosen, chosenProb = "", 0.0
    for index in range(len(chunks)):
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
# cat = select correct category [CATEGORY_MODE: topic = 18 GENERAL NEWS TOPICS (210-227), leaf = every leaf CODES.csv code, off = keep original]
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

def judgeSafe(item):
    index, digest, record, text = item
    global failedCalls
    delay = 2.0
    for attempt in range(MAX_RETRIES + 1):
        try:
            return judgeText(text, record.get("label", ""))
        except Exception as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", 0)
            if status in RETRY_STATUS and attempt < MAX_RETRIES:
                wait = delay + random.uniform(0, 1)
                try:
                    wait = max(wait, float(response.headers.get("Retry-After", 0)))
                except (TypeError, ValueError):
                    pass
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue
            with failedLock:
                failedCalls += 1
            print(f"judge failed on line {index}: {type(error).__name__}: {error}")
            return None

def processChunk(chunk, maxParallel = MAX_PARALLEL):
    hashes = loadHashes()
    claimed = set()
    kept = 0
    failed = []
    CLEAN_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    pending = []
    for index, record in chunk:
        text = record.get("text")
        if not isinstance(text, str):
            continue
        text = removeRedundantWhitespace(
            removeSpecialCharacters(removeHtmlTags(text))
        )
        if len(text.split()) < MIN_WORD_COUNT:
            continue
        digest = hashlib.sha256(text.lower().encode("utf-8")).hexdigest()
        if digest in hashes or digest in claimed:
            continue
        claimed.add(digest)
        pending.append((index, digest, record, text))

    with open(CLEAN_CORPUS_PATH, "a", encoding="utf-8") as out, ThreadPoolExecutor(maxParallel) as pool:
        for start in range(0, len(pending), maxParallel):
            batch = pending[start:start + maxParallel]
            for (index, digest, record, text), verdict in zip(batch, pool.map(judgeSafe, batch)):
                if verdict is None:
                    failed.append(index)
                    continue
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
            out.flush()

    if failed:
        with open(FAILED_PATH, "a", encoding="utf-8") as out:
            for index in failed:
                out.write(str(index) + "\n")
    return kept

if __name__ == "__main__":
    # load resume doc
    resumeLine = 0
    if os.path.exists(RESUME_PATH):
        resumeLine = int(open(RESUME_PATH).read().strip())

    processed = resumeLine
    print(f"category mode {CATEGORY_MODE} ({len(CATEGORY_CODES)} codes), model {JUDGE_MODELS[judgeModelIndex][0]}, parallel {MAX_PARALLEL}")
    # process chunks, resume from last processed line
    for chunk in readChunks(BATCH_SIZE, resumeLine):
        kept = processChunk(chunk, MAX_PARALLEL)
        processed += len(chunk)
        saveHashes()
        with open(RESUME_PATH, "w", encoding="utf-8") as f:
            f.write(str(processed))
        print(f"Kept {kept} records, processed {processed}, failed {failedCalls}, cost ${totalCost:.4f}")


