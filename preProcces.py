import os
import pandas as pd
from datasets import load_dataset
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import json
import logging
from tqdm import tqdm

RAW_DIR = '/media/user/2TB/raw_articles'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)

async def _fetch_one(session, sem, url):
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    content = await response.read()
                    encoding = response.charset or 'utf-8'
                    html = content.decode(encoding, errors='replace')
                    soup = BeautifulSoup(html, 'html.parser')
                    paragraphs = soup.find_all('p')
                    return ' '.join([para.get_text() for para in paragraphs])
        except Exception as e:
            print(f"Error fetching article from {url}: {e}")
    return None

async def _fetch_all(urls, concurrency=50):
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        async def fetch_and_update(url):
            return await _fetch_one(session, sem, url)
        return list(await asyncio.gather(*[fetch_and_update(url) for url in urls]))




def preProcess():
    total_processed = 0
    os.makedirs(RAW_DIR, exist_ok=True)
    codes_df = pd.read_csv('Codes.csv', header=None, names=['Code', 'Description'], dtype={'Code': str})

    codes = dict(zip(codes_df['Code'], codes_df['Description']))

    # logging.info("Loading dwb2023/gdelt-event-2025-v4 ...")
    # ds = load_dataset("dwb2023/gdelt-event-2025-v4", cache_dir='/media/user/2TB/huggingface_cache')
    # gdelt_raw_path = f'{RAW_DIR}/gdelt-event.jsonl'
    # already_fetched_gdelt = set()
    # if os.path.exists(gdelt_raw_path):
    #     with open(gdelt_raw_path) as f:
    #         for line in f:
    #             try:
    #                 already_fetched_gdelt.add(json.loads(line)['url'])
    #             except Exception:
    #                 pass
    # logging.info(f"gdelt-event: {len(already_fetched_gdelt)} URLs already saved, scanning for remaining ...")
    # pending_gdelt = []
    # for row in tqdm(ds['train'], desc='gdelt-event scan'):
    #     if row['SOURCEURL'] not in already_fetched_gdelt:
    #         pending_gdelt.append((row['SOURCEURL'], row['EventCode'], row['ActionGeo_CountryCode']))
    # logging.info(f"gdelt-event: {len(pending_gdelt)} URLs to fetch")
    # written = 0
    # with open(gdelt_raw_path, 'a') as raw_f, open('preprocessed_data.txt', 'a') as out_f:
    #     with tqdm(total=len(pending_gdelt), desc='gdelt-event', unit='url') as pbar:
    #         for i in range(0, len(pending_gdelt), 500):
    #             chunk = pending_gdelt[i:i + 500]
    #             texts = asyncio.run(_fetch_all([u for u, _, _ in chunk]))
    #             for (url, event_code, location), text in zip(chunk, texts):
    #                 raw_f.write(json.dumps({'url': url, 'text': text, 'EventCode': event_code, 'location': location}) + '\n')
    #                 if text:
    #                     label = codes.get(event_code, event_code)
    #                     out_f.write(str({'text': text, 'label': label, 'location': location}) + '\n')
    #                     written += 1
    #             raw_f.flush()
    #             out_f.flush()
    #             pbar.update(len(chunk))
    # total_processed += written
    # logging.info(f"gdelt-event: {written} new articles saved to {gdelt_raw_path} (total so far: {total_processed})")

    logging.info("Loading NEWS_CATEGORY.json ...")
    news_raw_path = f'{RAW_DIR}/news-category.jsonl'
    already_fetched_news = set()
    if os.path.exists(news_raw_path):
        with open(news_raw_path) as f:
            for line in f:
                try:
                    already_fetched_news.add(json.loads(line)['url'])
                except Exception:
                    pass
    news = []
    with open('NEWS_CATEGORY.json') as nf:
        for line in nf:
            line = line.strip()
            if line:
                try:
                    news.append(json.loads(line))
                except Exception:
                    pass
    all_news_items = [(item['link'], item['category']) for item in news if 'link' in item and 'category' in item]
    pending_news = [(url, cat) for url, cat in all_news_items if url not in already_fetched_news]
    logging.info(f"NEWS_CATEGORY: {len(already_fetched_news)} already saved, {len(pending_news)} remaining")
    written = 0
    with open(news_raw_path, 'a') as raw_f, open('preprocessed_data.txt', 'a') as out_f:
        with tqdm(total=len(pending_news), desc='NEWS_CATEGORY', unit='url') as pbar:
            for i in range(0, len(pending_news), 500):
                chunk = pending_news[i:i + 500]
                texts = asyncio.run(_fetch_all([u for u, _ in chunk]))
                for (url, category), text in zip(chunk, texts):
                    raw_f.write(json.dumps({'url': url, 'text': text, 'category': category}) + '\n')
                    if text:
                        out_f.write(str({'text': text, 'label': category, 'location': 'Unknown'}) + '\n')
                        written += 1
                raw_f.flush()
                out_f.flush()
                pbar.update(len(chunk))
    total_processed += written
    logging.info(f"NEWS_CATEGORY: {written} new articles saved to {news_raw_path} (total so far: {total_processed})")

    logging.info("Loading textminr/mn-ds ...")
    ds = load_dataset("textminr/mn-ds", cache_dir='/media/user/2TB/huggingface_cache')
    written = 0
    for row in tqdm(ds['train'], desc='textminr/mn-ds'):
        if "category_level_2" in row and "content" in row:
            category = row['category_level_2']
            article_text = row['content']
            sample = {'text': article_text, 'label': category, 'location': 'Unknown'}
            with open('preprocessed_data.txt', 'a') as f:
                f.write(str(sample) + '\n')
            written += 1
    total_processed += written
    logging.info(f"textminr/mn-ds: {written} written (total so far: {total_processed})")

    logging.info("Loading Alirezamp/news-category ...")
    ds = load_dataset("Alirezamp/news-category", cache_dir='/media/user/2TB/huggingface_cache')
    written = 0
    for row in tqdm(ds['train'], desc='Alirezamp/news-category'):
        if "text" in row and "category" in row:
            category = row['category']
            article_text = row['text']
            sample = {'text': article_text, 'label': category, 'location': 'Unknown'}
            with open('preprocessed_data.txt', 'a') as f:
                f.write(str(sample) + '\n')
            written += 1
    total_processed += written
    logging.info(f"Alirezamp/news-category: {written} written (total so far: {total_processed})")

    for idx in range(0, 8):
        subset = f"generate_text_classification_data_{idx}"
        logging.info(f"Loading sdiazlor/text-classification-news-topics [{subset}] ...")
        ds = load_dataset("sdiazlor/text-classification-news-topics", subset, cache_dir='/media/user/2TB/huggingface_cache')
        written = 0
        for row in tqdm(ds['train'], desc=f'sdiazlor subset {idx}'):
            if "input_text" in row and "label" in row:
                category = row['label']
                article_text = row['input_text']
                sample = {'text': article_text, 'label': category, 'location': 'Unknown'}
                with open('preprocessed_data.txt', 'a') as f:
                    f.write(str(sample) + '\n')
                written += 1
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
    ds = load_dataset("zeroshot/twitter-financial-news-topic", cache_dir='/media/user/2TB/huggingface_cache')
    written = 0
    for row in tqdm(ds['train'], desc='twitter-financial-news-topic'):
        if "text" in row and "label" in row:
            category = topics[str(row['label'])]
            article_text = row['text']
            sample = {'text': article_text, 'label': category, 'location': 'Unknown'}
            with open('preprocessed_data.txt', 'a') as f:
                f.write(str(sample) + '\n')
            written += 1
    total_processed += written
    logging.info(f"twitter-financial-news-topic: {written} written (total so far: {total_processed})")
    
    # logging.info("Loading stanford-oval/ccnews (2024) ...")
    # ds = load_dataset("stanford-oval/ccnews", "2024", cache_dir='/media/user/2TB/huggingface_cache')
    # written = 0
    # for row in tqdm(ds['train'], desc='ccnews/2024'):
    #     if "plain_text" in row and "categories" in row:
    #         category = row['categories'].split(',')[0].strip() if row['categories'] else 'Unknown'
    #         article_text = row['plain_text']
    #         sample = {'text': article_text, 'label': category, 'location': 'Unknown'}
    #         with open('preprocessed_data.txt', 'a') as f:
    #             f.write(str(sample) + '\n')
    #         written += 1
    # total_processed += written
    # logging.info(f"ccnews/2024: {written} written (total so far: {total_processed})")

    logging.info("Loading dwb2023/gdelt-gkg-march2020-v2 ...")
    ds = load_dataset("dwb2023/gdelt-gkg-march2020-v2", cache_dir='/media/user/2TB/huggingface_cache')
    gkg_raw_path = f'{RAW_DIR}/gdelt-gkg.jsonl'
    already_fetched_gkg = set()
    if os.path.exists(gkg_raw_path):
        with open(gkg_raw_path) as f:
            for line in f:
                try:
                    already_fetched_gkg.add(json.loads(line)['url'])
                except Exception:
                    pass
    logging.info(f"gdelt-gkg: {len(already_fetched_gkg)} URLs already saved, scanning for remaining ...")
    pending_gkg = []
    for row in tqdm(ds['train'], desc='gdelt-gkg scan'):
        if row.get('V2EnhancedLocations') and row.get('V2EnhancedThemes') and row.get('DocumentIdentifier'):
            if row['DocumentIdentifier'] not in already_fetched_gkg:
                pending_gkg.append((row['DocumentIdentifier'], row['V2EnhancedThemes'], row['V2EnhancedLocations']))
    logging.info(f"gdelt-gkg: {len(pending_gkg)} URLs to fetch")
    written = 0
    with open(gkg_raw_path, 'a') as raw_f, open('preprocessed_data.txt', 'a') as out_f:
        with tqdm(total=len(pending_gkg), desc='gdelt-gkg', unit='url') as pbar:
            for i in range(0, len(pending_gkg), 500):
                chunk = pending_gkg[i:i + 500]
                texts = asyncio.run(_fetch_all([u for u, _, _ in chunk]))
                for (url, themes, locations), text in zip(chunk, texts):
                    raw_f.write(json.dumps({'url': url, 'text': text, 'themes': themes, 'locations': locations}) + '\n')
                    if text:
                        category = themes.split(';')[0].split(',')[0]
                        loc_parts = locations.split(';')[0].split('#')
                        location = loc_parts[1] if len(loc_parts) > 1 else 'Unknown'
                        out_f.write(str({'text': text, 'label': category, 'location': location}) + '\n')
                        written += 1
                raw_f.flush()
                out_f.flush()
                pbar.update(len(chunk))
    total_processed += written
    logging.info(f"gdelt-gkg: {written} new articles saved to {gkg_raw_path} (total so far: {total_processed})")

    ag_news_labels = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    logging.info("Loading fancyzhx/ag_news ...")
    ds = load_dataset("fancyzhx/ag_news", cache_dir='/media/user/2TB/huggingface_cache')
    written = 0
    for row in tqdm(ds['train'], desc='ag_news'):
        if "text" in row and "label" in row:
            category = ag_news_labels.get(row['label'], str(row['label']))
            article_text = row['text']
            sample = {'text': article_text, 'label': category, 'location': 'Unknown'}
            with open('preprocessed_data.txt', 'a') as f:
                f.write(str(sample) + '\n')
            written += 1
    total_processed += written
    logging.info(f"ag_news: {written} written (total so far: {total_processed})")

    logging.info("Loading SetFit/bbc-news ...")
    ds = load_dataset("SetFit/bbc-news", cache_dir='/media/user/2TB/huggingface_cache')
    written = 0
    for row in tqdm(ds['train'], desc='bbc-news'):
        if "text" in row and "label_text" in row:
            category = row['label_text']
            article_text = row['text']
            sample = {'text': article_text, 'label': category, 'location': 'Unknown'}
            with open('preprocessed_data.txt', 'a') as f:
                f.write(str(sample) + '\n')
            written += 1
    total_processed += written
    logging.info(f"bbc-news: {written} written (total so far: {total_processed})")


    logging.info("Loading rjjan/reuters21578 (ModApte) ...")
    ds = load_dataset("parquet", data_files={"train": "hf://datasets/rjjan/reuters21578@refs/convert/parquet/ModApte/train/*.parquet"}, cache_dir='/media/user/2TB/huggingface_cache')
    written = 0
    for row in tqdm(ds['train'], desc='reuters/ModApte'):
        if "text" in row and "topics" in row and len(row['topics']) > 0:
            category = row['topics'][0]
            article_text = row['text']
            sample = {'text': article_text, 'label': category, 'location': 'Unknown'}
            with open('preprocessed_data.txt', 'a') as f:
                f.write(str(sample) + '\n')
            written += 1
    total_processed += written
    logging.info(f"reuters/ModApte: {written} written (total so far: {total_processed})")

    logging.info("Loading rjjan/reuters21578 (ModHayes) ...")
    ds = load_dataset("parquet", data_files={"train": "hf://datasets/rjjan/reuters21578@refs/convert/parquet/ModHayes/train/*.parquet"}, cache_dir='/media/user/2TB/huggingface_cache')
    written = 0
    for row in tqdm(ds['train'], desc='reuters/ModHayes'):
        if "text" in row and "topics" in row and len(row['topics']) > 0:
            category = row['topics'][0]
            article_text = row['text']
            sample = {'text': article_text, 'label': category, 'location': 'Unknown'}
            with open('preprocessed_data.txt', 'a') as f:
                f.write(str(sample) + '\n')
            written += 1
    total_processed += written
    logging.info(f"reuters/ModHayes: {written} written (total so far: {total_processed})")

    logging.info("Loading rjjan/reuters21578 (ModLewis) ...")
    ds = load_dataset("parquet", data_files={"train": "hf://datasets/rjjan/reuters21578@refs/convert/parquet/ModLewis/train/*.parquet"}, cache_dir='/media/user/2TB/huggingface_cache')
    written = 0
    for row in tqdm(ds['train'], desc='reuters/ModLewis'):
        if "text" in row and "topics" in row and len(row['topics']) > 0:
            category = row['topics'][0]
            article_text = row['text']
            sample = {'text': article_text, 'label': category, 'location': 'Unknown'}
            with open('preprocessed_data.txt', 'a') as f:
                f.write(str(sample) + '\n')
            written += 1
    total_processed += written
    logging.info(f"reuters/ModLewis: {written} written (total so far: {total_processed})")

    logging.info(f"=== DONE. Total samples written: {total_processed} ===")


if __name__ == "__main__":
    preProcess()
        


    

