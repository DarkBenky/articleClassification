"""Collect public news datasets into the shared preprocessed corpus."""

import asyncio
import json
import logging

import aiohttp
from bs4 import BeautifulSoup
from datasets import load_dataset
from tqdm import tqdm

from project_config import (
    CORPUS_PATH,
    HF_CACHE_DIR,
    NEWS_CATEGORY_PATH,
    RAW_ARTICLES_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


async def _fetch_one(session, sem, url):
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    content = await response.read()
                    encoding = response.charset or "utf-8"
                    html = content.decode(encoding, errors="replace")
                    soup = BeautifulSoup(html, "html.parser")
                    paragraphs = soup.find_all("p")
                    return " ".join(para.get_text() for para in paragraphs)
        except Exception as e:
            print(f"Error fetching article from {url}: {e}")
    return None


async def _fetch_all(urls, concurrency=50):
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        return list(await asyncio.gather(*[_fetch_one(session, sem, url) for url in urls]))


def _write_samples(samples, desc, total=None):
    """Append sample dicts to the corpus and return how many were written."""
    written = 0
    with open(CORPUS_PATH, "a", encoding="utf-8") as out_f, tqdm(samples, desc=desc, total=total) as bar:
        for sample in bar:
            out_f.write(str(sample) + "\n")
            written += 1
    return written


def preProcess():
    RAW_ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_processed = 0

    logging.info("Loading NEWS_CATEGORY.json ...")
    news_raw_path = RAW_ARTICLES_DIR / "news-category.jsonl"
    already_fetched_news = set()
    if news_raw_path.exists():
        with open(news_raw_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    already_fetched_news.add(json.loads(line)["url"])
                except Exception:
                    pass

    news = []
    with open(NEWS_CATEGORY_PATH, "r", encoding="utf-8") as nf:
        for line in nf:
            line = line.strip()
            if line:
                try:
                    news.append(json.loads(line))
                except Exception:
                    pass

    all_news_items = [(item["link"], item["category"]) for item in news if "link" in item and "category" in item]
    pending_news = [(url, cat) for url, cat in all_news_items if url not in already_fetched_news]
    logging.info(f"NEWS_CATEGORY: {len(already_fetched_news)} already saved, {len(pending_news)} remaining")

    written = 0
    with open(news_raw_path, "a", encoding="utf-8") as raw_f, open(CORPUS_PATH, "a", encoding="utf-8") as out_f:
        with tqdm(total=len(pending_news), desc="NEWS_CATEGORY", unit="url") as pbar:
            for i in range(0, len(pending_news), 500):
                chunk = pending_news[i:i + 500]
                texts = asyncio.run(_fetch_all([u for u, _ in chunk]))
                for (url, category), text in zip(chunk, texts):
                    raw_f.write(json.dumps({"url": url, "text": text, "category": category}) + "\n")
                    if text:
                        out_f.write(str({"text": text, "label": category, "location": "Unknown"}) + "\n")
                        written += 1
                raw_f.flush()
                out_f.flush()
                pbar.update(len(chunk))
    total_processed += written
    logging.info(f"NEWS_CATEGORY: {written} new articles saved to {news_raw_path} (total so far: {total_processed})")

    logging.info("Loading textminr/mn-ds ...")
    ds = load_dataset("textminr/mn-ds", cache_dir=str(HF_CACHE_DIR))
    records = (
        {"text": row["content"], "label": row["category_level_2"], "location": "Unknown"}
        for row in ds["train"]
        if "category_level_2" in row and "content" in row
    )
    written = _write_samples(records, "textminr/mn-ds", total=len(ds["train"]))
    total_processed += written
    logging.info(f"textminr/mn-ds: {written} written (total so far: {total_processed})")

    logging.info("Loading Alirezamp/news-category ...")
    ds = load_dataset("Alirezamp/news-category", cache_dir=str(HF_CACHE_DIR))
    records = (
        {"text": row["text"], "label": row["category"], "location": "Unknown"}
        for row in ds["train"]
        if "text" in row and "category" in row
    )
    written = _write_samples(records, "Alirezamp/news-category", total=len(ds["train"]))
    total_processed += written
    logging.info(f"Alirezamp/news-category: {written} written (total so far: {total_processed})")

    for idx in range(8):
        subset = f"generate_text_classification_data_{idx}"
        logging.info(f"Loading sdiazlor/text-classification-news-topics [{subset}] ...")
        ds = load_dataset("sdiazlor/text-classification-news-topics", subset, cache_dir=str(HF_CACHE_DIR))
        records = (
            {"text": row["input_text"], "label": row["label"], "location": "Unknown"}
            for row in ds["train"]
            if "input_text" in row and "label" in row
        )
        written = _write_samples(records, f"sdiazlor subset {idx}", total=len(ds["train"]))
        total_processed += written
        logging.info(f"sdiazlor subset {idx}: {written} written (total so far: {total_processed})")

    topics = {
        "0": "Analyst Update",
        "1": "Fed | Central Banks",
        "2": "Company | Product News",
        "3": "Treasuries | Corporate Debt",
        "4": "Dividend",
        "5": "Earnings",
        "6": "Energy | Oil",
        "7": "Financiers",
        "8": "Currencies",
        "9": "General News | Opinion",
        "10": "Gold | Metals | Materials",
        "11": "IPO",
        "12": "Legal | Regulation",
        "13": "M&A | Investments",
        "14": "Macro",
        "15": "Markets",
        "16": "Politics",
        "17": "Personnel Change",
        "18": "Stock Commentary",
        "19": "Stock Movement",
    }

    logging.info("Loading zeroshot/twitter-financial-news-topic ...")
    ds = load_dataset("zeroshot/twitter-financial-news-topic", cache_dir=str(HF_CACHE_DIR))
    records = (
        {"text": row["text"], "label": topics[str(row["label"])], "location": "Unknown"}
        for row in ds["train"]
        if "text" in row and "label" in row
    )
    written = _write_samples(records, "twitter-financial-news-topic", total=len(ds["train"]))
    total_processed += written
    logging.info(f"twitter-financial-news-topic: {written} written (total so far: {total_processed})")

    logging.info("Loading dwb2023/gdelt-gkg-march2020-v2 ...")
    ds = load_dataset("dwb2023/gdelt-gkg-march2020-v2", cache_dir=str(HF_CACHE_DIR))
    gkg_raw_path = RAW_ARTICLES_DIR / "gdelt-gkg.jsonl"
    already_fetched_gkg = set()
    if gkg_raw_path.exists():
        with open(gkg_raw_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    already_fetched_gkg.add(json.loads(line)["url"])
                except Exception:
                    pass

    logging.info(f"gdelt-gkg: {len(already_fetched_gkg)} URLs already saved, scanning for remaining ...")
    pending_gkg = []
    for row in tqdm(ds["train"], desc="gdelt-gkg scan"):
        if row.get("V2EnhancedLocations") and row.get("V2EnhancedThemes") and row.get("DocumentIdentifier"):
            if row["DocumentIdentifier"] not in already_fetched_gkg:
                pending_gkg.append((row["DocumentIdentifier"], row["V2EnhancedThemes"], row["V2EnhancedLocations"]))
    logging.info(f"gdelt-gkg: {len(pending_gkg)} URLs to fetch")

    written = 0
    with open(gkg_raw_path, "a", encoding="utf-8") as raw_f, open(CORPUS_PATH, "a", encoding="utf-8") as out_f:
        with tqdm(total=len(pending_gkg), desc="gdelt-gkg", unit="url") as pbar:
            for i in range(0, len(pending_gkg), 500):
                chunk = pending_gkg[i:i + 500]
                texts = asyncio.run(_fetch_all([u for u, _, _ in chunk]))
                for (url, themes, locations), text in zip(chunk, texts):
                    raw_f.write(json.dumps({"url": url, "text": text, "themes": themes, "locations": locations}) + "\n")
                    if text:
                        category = themes.split(";")[0].split(",")[0]
                        loc_parts = locations.split(";")[0].split("#")
                        location = loc_parts[1] if len(loc_parts) > 1 else "Unknown"
                        out_f.write(str({"text": text, "label": category, "location": location}) + "\n")
                        written += 1
                raw_f.flush()
                out_f.flush()
                pbar.update(len(chunk))
    total_processed += written
    logging.info(f"gdelt-gkg: {written} new articles saved to {gkg_raw_path} (total so far: {total_processed})")

    ag_news_labels = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    logging.info("Loading fancyzhx/ag_news ...")
    ds = load_dataset("fancyzhx/ag_news", cache_dir=str(HF_CACHE_DIR))
    records = (
        {"text": row["text"], "label": ag_news_labels.get(row["label"], str(row["label"])), "location": "Unknown"}
        for row in ds["train"]
        if "text" in row and "label" in row
    )
    written = _write_samples(records, "ag_news", total=len(ds["train"]))
    total_processed += written
    logging.info(f"ag_news: {written} written (total so far: {total_processed})")

    logging.info("Loading SetFit/bbc-news ...")
    ds = load_dataset("SetFit/bbc-news", cache_dir=str(HF_CACHE_DIR))
    records = (
        {"text": row["text"], "label": row["label_text"], "location": "Unknown"}
        for row in ds["train"]
        if "text" in row and "label_text" in row
    )
    written = _write_samples(records, "bbc-news", total=len(ds["train"]))
    total_processed += written
    logging.info(f"bbc-news: {written} written (total so far: {total_processed})")

    for subset in ("ModApte", "ModHayes", "ModLewis"):
        logging.info(f"Loading rjjan/reuters21578 ({subset}) ...")
        ds = load_dataset(
            "parquet",
            data_files={"train": f"hf://datasets/rjjan/reuters21578@refs/convert/parquet/{subset}/train/*.parquet"},
            cache_dir=str(HF_CACHE_DIR),
        )
        records = (
            {"text": row["text"], "label": row["topics"][0], "location": "Unknown"}
            for row in ds["train"]
            if row.get("text") and row.get("topics")
        )
        written = _write_samples(records, f"reuters/{subset}", total=len(ds["train"]))
        total_processed += written
        logging.info(f"reuters/{subset}: {written} written (total so far: {total_processed})")

    logging.info(f"=== DONE. Total samples written: {total_processed} ===")


if __name__ == "__main__":
    preProcess()
