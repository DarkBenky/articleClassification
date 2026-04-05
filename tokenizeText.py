from locationModel import CONTEXT_SIZE
import ast
import json
import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer

DATA_PATH = "/media/user/2TB/preprocessed_data.txt"
OUT_DIR   = "/media/user/2TB/tokenizedtext"
CHUNK_SIZE = 512  # items held in RAM at once

_tokenizer = None

def _init_tokenizer():
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def _tokenize_item(args):
    text, label = args
    result = _tokenizer(text, padding="max_length", truncation=True, max_length=CONTEXT_SIZE, return_tensors="np")
    return result["input_ids"][0].astype("int32"), label

def _iter_data(location_to_idx):
    """Yield (text, label) one line at a time — never loads full file into RAM."""
    with open(DATA_PATH, "r") as f:
        for line in f:
            item = ast.literal_eval(line.strip())
            yield item["text"], location_to_idx[item["location"]]

def _chunked(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

if __name__ == "__main__":
    with open("unique_locations.json", "r") as f:
        unique_locations = json.load(f)

    location_to_idx = {loc: idx for idx, loc in enumerate(unique_locations.keys())}

    # Single pass to count lines (cheap — no parsing)
    print("Counting total items...")
    with open(DATA_PATH, "r") as f:
        total = sum(1 for _ in f)
    print(f"Total items: {total}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Pre-allocate memory-mapped arrays on disk; only one CHUNK_SIZE slice lives in RAM
    X_mmap = np.memmap(os.path.join(OUT_DIR, "X.dat"), dtype="int32",  mode="w+", shape=(total, CONTEXT_SIZE))
    y_mmap = np.memmap(os.path.join(OUT_DIR, "y.dat"), dtype="int64",  mode="w+", shape=(total,))

    idx = 0
    start_time = time.time()
    chunk_count = 0
    total_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"Starting tokenization with 8 workers | chunk size: {CHUNK_SIZE} | total chunks: {total_chunks}")
    print("-" * 60)

    with ProcessPoolExecutor(max_workers=8, initializer=_init_tokenizer) as executor:
        for chunk in _chunked(_iter_data(location_to_idx), CHUNK_SIZE):
            chunk_start = time.time()
            results = list(executor.map(_tokenize_item, chunk, chunksize=32))
            for input_ids, label in results:
                X_mmap[idx] = input_ids
                y_mmap[idx] = label
                idx += 1
            X_mmap.flush()
            y_mmap.flush()
            chunk_count += 1

            elapsed = time.time() - start_time
            chunk_elapsed = time.time() - chunk_start
            pct = idx / total * 100
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (total - idx) / rate if rate > 0 else 0
            print(
                f"[{chunk_count}/{total_chunks}] {idx}/{total} ({pct:.1f}%) | "
                f"chunk: {chunk_elapsed:.1f}s | elapsed: {elapsed:.1f}s | "
                f"rate: {rate:.0f} items/s | ETA: {eta:.0f}s"
            )

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Finished {total} items in {total_time:.1f}s ({total/total_time:.0f} items/s)")

    # Save shape metadata so the arrays can be reloaded later:
    #   X = np.memmap('X.dat', dtype='int32', mode='r', shape=(total, CONTEXT_SIZE))
    #   y = np.memmap('y.dat', dtype='int64', mode='r', shape=(total,))
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump({"total": total, "context_size": CONTEXT_SIZE}, f)

    print(f"Done. Saved to {OUT_DIR}")
    