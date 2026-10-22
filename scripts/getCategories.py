"""Count category and location labels in the preprocessed corpus."""

import ast
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from project_config import (
    ARTIFACTS_DIR,
    CORPUS_PATH,
    PROCESSED_DATA_DIR,
    UNIQUE_CATEGORIES_PATH,
    UNIQUE_LOCATIONS_PATH,
)

if __name__ == "__main__":
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    unique_categories = Counter()
    unique_locations = Counter()

    count = 0
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for count, line in enumerate(f, start=1):
            data = ast.literal_eval(line)
            unique_categories[data["label"]] += 1
            unique_locations[data["location"]] += 1
            if count % 1000 == 0:
                print(f"Processed {count:,} lines", end="\r")
    print(f"Processed {count:,} lines")
    
    # save unique categories and locations to json files
    with open(UNIQUE_CATEGORIES_PATH, "w", encoding="utf-8") as f:
        json.dump(unique_categories, f)
    with open(UNIQUE_LOCATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(unique_locations, f)
    
    # show top 20 categories
    sorted_categories = sorted(unique_categories.items(), key=lambda x: x[1], reverse=True)[:20]
    categories, counts = zip(*sorted_categories)
    plt.bar(categories, counts)
    plt.xticks(rotation=90)
    plt.title("Top 20 Categories")
    plt.xlabel("Categories")
    plt.ylabel("Counts")
    # save figure
    plt.savefig(ARTIFACTS_DIR / "top_categories.png", dpi=150)
    plt.close()

    # show top 20 locations
    filtered_locations = {k: v for k, v in unique_locations.items() if k is not None}
    sorted_locations = sorted(filtered_locations.items(), key=lambda x: x[1], reverse=True)[:20]
    locations, counts = zip(*sorted_locations)
    plt.bar(locations, counts)
    plt.xticks(rotation=90)
    plt.title("Top 20 Locations")
    plt.xlabel("Locations")
    plt.ylabel("Counts")
    # save figure
    plt.savefig(ARTIFACTS_DIR / "raw_top_locations.png", dpi=150)
    plt.close()
    