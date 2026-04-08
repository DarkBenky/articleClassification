import os

DATA_PATH = "/media/user/2TB/preprocessed_data.txt"


def addToDataset(article: dict):
    """
    article = {"text": str, "location": str (FIPS or ISO code), "category": str}
    """
    record = {
        "text": article["text"],
        "label": article["category"],
        "location": article["location"],
    }
    with open(DATA_PATH, "a", encoding="utf-8") as f:
        f.write(repr(record) + "\n")
    print(f"Appended: [{record['location']}] {record['text'][:60]}...")


if __name__ == "__main__":
    addToDataset({
        "text": "The Federal Reserve raised interest rates by 25 basis points amid persistent inflation across the United States.",
        "location": "US",
        "category": "Economy",
    })

    addToDataset({
        "text": "Flooding in southern Germany has displaced thousands as the Rhine river reaches record levels.",
        "location": "GM",
        "category": "Natural disaster",
    })

    addToDataset({
        "text": "Japan's parliament approved a new defense budget doubling military spending over the next five years.",
        "location": "JA",
        "category": "Defense",
    })
