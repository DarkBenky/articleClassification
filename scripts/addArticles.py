"""Append article records to the shared corpus file."""

from project_config import CORPUS_PATH


def addToDataset(article: dict, path = None):
    """Append one {"text", "location", "category"} record to the corpus."""
    if not path:
        path = CORPUS_PATH
    
    record = {
        "text": article["text"],
        "label": article["category"],
        "location": article["location"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(repr(record) + "\n")
    model_tag = f"[{article.get('model', '?')}] " if article.get("model") else ""
    print(f"Appended: [{record['location']}] {model_tag}{record['text'][:60]}...")
