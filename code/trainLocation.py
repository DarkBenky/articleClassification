import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import ast
import csv
import json
import random
import uuid
from itertools import islice
from pprint import pprint

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import wandb

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")),
)
from project_config import (
    CORPUS_PATH,
    FIPS_CODES_PATH,
    LOCATION_TO_FIPS_PATH,
    MODELS_DIR,
)

TRAIN_HEAD = True
LEARNING_RATE_HEAD = 1e-4
TRAIN_BASE = False
LEARNING_RATE_BASE = 1e-5
BATCH_SIZE = 32
EPOCHS = 3
CHUNK_SIZE = 4096
MAX_LENGTH = 512
LOG_EVERY = 10
DROPOUT = 0.2
CHECKPOINT_EVERY = 100

if torch.cuda.is_available():
    deviceIndex = max(
        range(torch.cuda.device_count()),
        key=lambda index: torch.cuda.mem_get_info(index)[0],
    )
    device = torch.device(f"cuda:{deviceIndex}")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


def getFipCodes():
    with open(FIPS_CODES_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fipCodes = {}
        for row in reader:
            fips = row["FIPS 10-4"].strip()
            countryName = row["Name"].strip()
            if fips and fips != "-":
                fipCodes[fips] = countryName
    return list(fipCodes.keys()), list(fipCodes.values())


def countLines(path):
    total = 0
    lastByte = b""
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            total += chunk.count(b"\n")
            lastByte = chunk[-1:]
    return total + (1 if lastByte and lastByte != b"\n" else 0)


fipCodes, countryNames = getFipCodes()
fipCodeCount = len(fipCodes)
print(f"FIP codes count: {fipCodeCount}")

labelToIndex = {name: i for i, name in enumerate(countryNames)}
labelToIndex.update({code: i for i, code in enumerate(fipCodes)})

with open(LOCATION_TO_FIPS_PATH, encoding="utf-8") as f:
    locationToFips = json.load(f)

print(f"Counting records in {CORPUS_PATH}...")
totalRecords = countLines(CORPUS_PATH)
fullChunks, remainder = divmod(totalRecords, CHUNK_SIZE)
stepsPerEpoch = fullChunks * ((CHUNK_SIZE + BATCH_SIZE - 1) // BATCH_SIZE)
if remainder:
    stepsPerEpoch += (remainder + BATCH_SIZE - 1) // BATCH_SIZE
totalSteps = stepsPerEpoch * EPOCHS
print(
    f"Corpus records: {totalRecords:,} | steps per epoch: {stepsPerEpoch:,} | "
    f"total steps: {totalSteps:,}"
)

name = "Qwen/Qwen3-Embedding-0.6B"
tok = AutoTokenizer.from_pretrained(name, padding_side="left")
base = AutoModel.from_pretrained(name, torch_dtype=torch.float32).to(device)
size = base.config.hidden_size

head = nn.Sequential(
    nn.LayerNorm(size),
    nn.Linear(size, 4096),
    nn.GELU(),
    nn.Dropout(DROPOUT),
    nn.Linear(4096, 1024),
    nn.GELU(),
    nn.Dropout(DROPOUT),
    nn.Linear(1024, 512),
    nn.GELU(),
    nn.Dropout(DROPOUT),
    nn.Linear(512, fipCodeCount),
).to(device)

for p in base.parameters():
    p.requires_grad = TRAIN_BASE
for p in head.parameters():
    p.requires_grad = TRAIN_HEAD

totalParams = sum(p.numel() for p in head.parameters())
trainableParams = sum(p.numel() for p in head.parameters() if p.requires_grad)
print(f"Head parameters: {totalParams:,} total, {trainableParams:,} trainable")


def runModel(texts):
    x = tok(texts, padding=True, truncation=True, max_length=MAX_LENGTH,
            return_tensors="pt").to(device)
    h = base(**x).last_hidden_state[:, -1]
    return head(h)


def resolvePredictionToLocationNames(predictions, topK=5):
    predictions = torch.softmax(predictions.float(), dim=-1)
    results = []
    for prediction in predictions:
        prediction = prediction.tolist()
        top = sorted(range(len(prediction)), key=lambda i: prediction[i], reverse=True)[:topK]
        results.append({countryNames[i]: prediction[i] for i in top})
    return results


def readChunks(chunkSize):
    with open(CORPUS_PATH, encoding="utf-8") as f:
        while True:
            lines = list(islice(f, chunkSize))
            if not lines:
                return
            records = [ast.literal_eval(line) for line in lines]
            random.shuffle(records)
            yield records


def batches(chunkSize, batchSize):
    for chunk in readChunks(chunkSize):
        for i in range(0, len(chunk), batchSize):
            yield chunk[i:i + batchSize]


lossFn = nn.CrossEntropyLoss()

base.config.use_cache = False

groups = []
if TRAIN_HEAD:
    groups.append({"params": list(head.parameters()), "lr": LEARNING_RATE_HEAD})
if TRAIN_BASE:
    groups.append({"params": list(base.parameters()), "lr": LEARNING_RATE_BASE})
    base.gradient_checkpointing_enable()
allParams = [p for g in groups for p in g["params"]]
optimizer = torch.optim.AdamW(groups)


def trainStep(batch):
    rows = []
    for record in batch:
        fips = locationToFips.get(record["location"], record["location"])
        if fips in labelToIndex:
            rows.append((record["text"], labelToIndex[fips]))
    if not rows:
        return None
    y = torch.tensor([label for _, label in rows], device=device)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        out = runModel([text for text, _ in rows])
    logits = out.float()
    loss = lossFn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(allParams, 1.0)
    optimizer.step()
    probs = torch.softmax(logits, dim=-1)
    predictions = probs.argmax(-1)
    acc = (predictions == y).float().mean().item()
    sample = (
        rows[0][0],
        y[0].item(),
        predictions[0].item(),
        probs[0, predictions[0]].item(),
    )
    return loss.item(), acc, len(rows), sample

run_name = uuid.uuid4().hex[:8]

wandb.init(
    project="article-location-classifier",
    name=run_name,
    config={
        "run_name": run_name,
        "total_params": totalParams,
        "trainable_params": trainableParams,
        "learning_rate_head": LEARNING_RATE_HEAD,
        "train_head": TRAIN_HEAD,
        "learning_rate_base": LEARNING_RATE_BASE,
        "train_base": TRAIN_BASE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "chunk_size": CHUNK_SIZE,
        "dropout": DROPOUT,
        "checkpoint_every": CHECKPOINT_EVERY,
        "total_records": totalRecords,
        "total_steps": totalSteps,
    },
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
bestModelPath = MODELS_DIR / f"location_classifier_torch_{run_name}.pt"
bestWindowLoss = float("inf")
windowLoss = 0.0
windowSteps = 0
sampleTable = wandb.Table(
    columns=["step", "epoch", "text", "expected", "predicted", "confidence"],
    log_mode="MUTABLE",
)
saveTable = wandb.Table(
    columns=["step", "epoch", "window_loss"],
    log_mode="MUTABLE",
)


def removeOldCheckpoints(keepPath):
    for path in MODELS_DIR.glob("location_classifier_torch_*.pt"):
        if path != keepPath:
            path.unlink(missing_ok=True)
    for path in MODELS_DIR.glob("location_classifier_torch*.tmp"):
        path.unlink(missing_ok=True)
    legacyPath = MODELS_DIR / "location_classifier_torch.pt"
    if legacyPath != keepPath and legacyPath.exists():
        legacyPath.unlink()


def saveIfBest(avgLoss, epoch, step, currentBest):
    if avgLoss >= currentBest:
        return currentBest, False
    checkpoint = {
        "head": head.state_dict(),
        "fipCodes": fipCodes,
        "countryNames": countryNames,
        "base_name": name,
        "train_base": TRAIN_BASE,
        "run_name": run_name,
        "epoch": epoch,
        "step": step,
        "loss": avgLoss,
    }
    if TRAIN_BASE:
        checkpoint["base"] = base.state_dict()
    tempPath = bestModelPath.with_suffix(".tmp")
    torch.save(checkpoint, tempPath)
    os.replace(tempPath, bestModelPath)
    removeOldCheckpoints(bestModelPath)
    return avgLoss, True


base.train(TRAIN_BASE)
head.train(TRAIN_HEAD)
step = 0
skipped = 0
seen = 0

for epoch in range(EPOCHS):
    epochLoss = 0.0
    epochBatches = 0
    for batch in batches(CHUNK_SIZE, BATCH_SIZE):
        seen += len(batch)
        res = trainStep(batch)
        if res is None:
            skipped += len(batch)
            continue
        loss, acc, used, sample = res
        skipped += len(batch) - used
        step += 1
        epochLoss += loss
        epochBatches += 1
        windowLoss += loss
        windowSteps += 1
        skipRate = skipped / seen * 100 if seen else 0.0
        progressPct = step / totalSteps * 100 if totalSteps else 0.0
        logData = {
            "loss": loss,
            "acc": acc,
            "epoch": epoch,
            "skip_pct": skipRate,
            "progress_pct": progressPct,
        }
        if step % LOG_EVERY == 0:
            text, expected, predicted, confidence = sample
            preview = " ".join(text.split())[:500]
            expectedLabel = f"{countryNames[expected]} ({fipCodes[expected]})"
            predictedLabel = f"{countryNames[predicted]} ({fipCodes[predicted]})"
            sampleTable.add_data(step, epoch, preview, expectedLabel, predictedLabel, confidence)
            logData["sample_predictions"] = sampleTable
            print(
                f"epoch {epoch} step {step}/{totalSteps} ({progressPct:.1f}%) "
                f"loss {loss:.4f} acc {acc:.3f} skip {skipRate:.1f}%\n"
                f"  expected {expectedLabel} | predicted {predictedLabel} {confidence:.3f}\n"
                f"  text {preview[:100]}"
            )
        if step % CHECKPOINT_EVERY == 0 and windowSteps:
            windowAvg = windowLoss / windowSteps
            bestWindowLoss, saved = saveIfBest(windowAvg, epoch, step, bestWindowLoss)
            logData["window_loss"] = windowAvg
            logData["best_window_loss"] = bestWindowLoss
            logData["checkpoint_saved"] = int(saved)
            if saved:
                saveTable.add_data(step, epoch, windowAvg)
                logData["checkpoint_saves"] = saveTable
                logData["saved_step"] = step
                print(f"step {step}/{totalSteps} window loss {windowAvg:.4f} saved {bestModelPath}")
            windowLoss = 0.0
            windowSteps = 0
        wandb.log(logData, step=step)
    if epochBatches == 0:
        continue
    epochLoss /= epochBatches
    print(f"epoch {epoch} avg loss {epochLoss:.4f}")

if windowSteps:
    windowAvg = windowLoss / windowSteps
    bestWindowLoss, saved = saveIfBest(windowAvg, epoch, step, bestWindowLoss)
    finalLog = {
        "final_window_loss": windowAvg,
        "best_window_loss": bestWindowLoss,
        "checkpoint_saved": int(saved),
    }
    if saved:
        saveTable.add_data(step, epoch, windowAvg)
        finalLog["checkpoint_saves"] = saveTable
        finalLog["saved_step"] = step
        print(f"step {step}/{totalSteps} final window loss {windowAvg:.4f} saved {bestModelPath}")
    wandb.log(finalLog, step=step)

base.eval()
head.eval()
with torch.no_grad():
    pprint(resolvePredictionToLocationNames(runModel(["hello world", "test"]), topK=5))

wandb.finish()