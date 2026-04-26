from locationModel import CONTEXT_SIZE
from groupLocations import flipCodeToName
import ast
import json
import os
import random
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    try:
        result = _tokenizer(text, padding="max_length", truncation=True, max_length=CONTEXT_SIZE, return_tensors="np")
        return result["input_ids"][0].astype("int32"), label
    except Exception:
        return None, None

def _iter_data(location_to_fips, fips_to_idx, lines_read_counter):
    """Yield (text, label) one line at a time — never loads full file into RAM."""
    with open(DATA_PATH, "r") as f:
        for line in f:
            lines_read_counter[0] += 1
            try:
                item = ast.literal_eval(line.strip())
                text = item.get("text")
                if not isinstance(text, str) or not text.strip() or item["location"] is None or item["location"] not in location_to_fips:
                    continue
                fips = location_to_fips[item["location"]]
                if fips not in fips_to_idx:
                    continue
                if fips == "Unknown" and random.random() > 0.0005:  # Keep only ~0.05% of Unknown to avoid dominating training
                    continue
                yield text, fips_to_idx[fips]
            except Exception:
                continue

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
    with open("location_to_fips.json", "r") as f:
        location_to_fips = json.load(f)

    with open("unique_fips_locations.json", "r") as f:
        fips_locations = json.load(f)

    # Index is determined by position in the reduced FIPS label set
    fips_to_idx = {fips: idx for idx, fips in enumerate(fips_locations.keys())}

    # Compose: original location string → FIPS code → integer label
    location_to_idx = {
        loc: fips_to_idx[fips]
        for loc, fips in location_to_fips.items()
        if fips in fips_to_idx
    }

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
    lines_read = [0]  # mutable counter shared with _iter_data
    start_time = time.time()

    print(f"Starting tokenization with 24 workers | chunk size: {CHUNK_SIZE} | total lines: {total}")
    print("-" * 60)

    with ProcessPoolExecutor(max_workers=24, initializer=_init_tokenizer) as executor:
        for chunk in _chunked(_iter_data(location_to_fips, fips_to_idx, lines_read), CHUNK_SIZE):
            chunk_start = time.time()
            try:
                results = list(executor.map(_tokenize_item, chunk, chunksize=32))
            except Exception:
                results = []
            for input_ids, label in results:
                if input_ids is None:
                    continue
                X_mmap[idx] = input_ids
                y_mmap[idx] = label
                idx += 1
            X_mmap.flush()
            y_mmap.flush()

            elapsed = time.time() - start_time
            chunk_elapsed = time.time() - chunk_start
            lines_pct = lines_read[0] / total * 100
            rate = lines_read[0] / elapsed if elapsed > 0 else 0
            eta = (total - lines_read[0]) / rate if rate > 0 else 0
            print(
                f"[{lines_read[0]}/{total} lines ({lines_pct:.1f}%)] written: {idx} | "
                f"chunk: {chunk_elapsed:.1f}s | elapsed: {elapsed:.1f}s | "
                f"rate: {rate:.0f} lines/s | ETA: {eta:.0f}s"
            )

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Finished. Read {total} lines, wrote {idx} valid items in {total_time:.1f}s ({total/total_time:.0f} lines/s)")
    print(f"Kept {idx/total*100:.1f}% of all lines ({idx} items)")

    # Save shape metadata so the arrays can be reloaded later:
    #   X = np.memmap('X.dat', dtype='int32', mode='r', shape=(total, CONTEXT_SIZE))
    #   y = np.memmap('y.dat', dtype='int64', mode='r', shape=(total,))
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump({"total": total, "valid_total": idx, "context_size": CONTEXT_SIZE}, f)

    label_counts = np.bincount(y_mmap[:idx].astype(np.int64), minlength=len(fips_to_idx))
    idx_to_fips  = {v: k for k, v in fips_to_idx.items()}

    top_n   = 30
    order   = np.argsort(label_counts)[::-1][:top_n]
    top_codes  = [idx_to_fips[i] for i in order]
    top_counts = [int(label_counts[i]) for i in order]

    print("\nTop categories in tokenized dataset:")
    print(f"  {'FIPS':<6} {'Count':>10}  Name")
    print("  " + "-" * 40)
    for code, count in zip(top_codes, top_counts):
        name = flipCodeToName.get(code, code)
        pct  = count / idx * 100 if idx else 0
        print(f"  {code:<6} {count:>10,}  {name} ({pct:.1f}%)")

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(top_codes[::-1], top_counts[::-1], color="steelblue")
    ax.bar_label(bars, labels=[f"{c:,}" for c in top_counts[::-1]], padding=4, fontsize=8)
    ax.set_xlabel("Sample count")
    ax.set_title(f"Top {top_n} locations in tokenized dataset ({idx:,} total samples)")
    ax.margins(x=0.12)
    plt.tight_layout()
    chart_path = "top_locations.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"\nChart saved to {chart_path}")

    print(f"Done. Saved to {OUT_DIR}")