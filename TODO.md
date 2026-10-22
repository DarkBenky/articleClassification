# TODO

- [ ] Process all news from GDELT Events (<http://data.gdeltproject.org/events/index.html>)
    - The old loader was removed during cleanup; reimplement it using `data/raw/CODES.csv` for CAMEO labels.

- [ ] Categories are now mass consolidated to roughly +/-100
    - `scripts/consolidateCategories.py` maps category labels to GENERAL NEWS TOPICS (210-227).

- [ ] Train a small embedding model for this task, for example:

```python
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

name = "Qwen/Qwen3-Embedding-0.6B"
texts = ["Berlin raised rates today.", "Tokyo won the match 2-1."] * 50
labels = ["GM", "JA"] * 50

codes = sorted(set(labels))
code2id = {c: i for i, c in enumerate(codes)}

tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(codes))
model.config.pad_token_id = tok.pad_token_id

def prep(batch):
    out = tok(batch["text"], truncation=True, max_length=512)
    out["labels"] = [code2id[l] for l in batch["label"]]
    return out

ds = Dataset.from_dict({"text": texts, "label": labels}).map(prep, batched=True)

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=2,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    report_to="none",
)

Trainer(model=model, args=args, train_dataset=ds, tokenizer=tok).train()
model.save_pretrained("countryClf")
tok.save_pretrained("countryClf")
```
