"""Resolve dataset category labels to the classes used by training.

The training scripts index classes with every name and code in CODES.csv and
prefer ``category_to_cameo.json`` when a raw label is present there.  This
module exposes that same resolution so data producers (``generateArticles.py``)
can verify their labels up front instead of discovering later that training
silently skipped them.
"""

import csv
import json

from project_config import CAMEO_CODES_PATH, CATEGORY_TO_CAMEO_PATH

NONE_KEY = "none"


def loadCameoCodes():
    """Return parallel (codes, names) lists for the two-column CODES.csv rows."""
    codes, names = [], []
    with open(CAMEO_CODES_PATH, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) != 2:
                continue
            code, name = row[0].strip(), row[1].strip()
            if code and name:
                codes.append(code)
                names.append(name)
    return codes, names


def loadCategoryToCameo():
    with open(CATEGORY_TO_CAMEO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def buildLabelIndex(codes, names):
    """Map both code and name spellings to a class index, as training does."""
    index = {name: i for i, name in enumerate(names)}
    index.update({code: i for i, code in enumerate(codes)})
    return index


def resolveCategoryCode(label, categoryToCameo):
    """Return the canonical code for a raw label, or None when unusable."""
    code = categoryToCameo.get(label, label)
    return None if code == NONE_KEY else code


def resolveCategoryIndex(label, categoryToCameo, labelToIndex):
    """Return the class index for a raw label, or None when training skips it."""
    code = resolveCategoryCode(label, categoryToCameo)
    return None if code is None else labelToIndex.get(code)


def validateCategories(labels, source="labels"):
    """Fail fast when a label would be dropped by the training scripts."""
    codes, names = loadCameoCodes()
    labelToIndex = buildLabelIndex(codes, names)
    categoryToCameo = loadCategoryToCameo()

    unresolved = [
        label
        for label in labels
        if resolveCategoryIndex(label, categoryToCameo, labelToIndex) is None
    ]
    if unresolved:
        raise SystemExit(
            f"{len(unresolved)} of {len(labels)} {source} have no training class, "
            f"so training would drop every record labelled with them:\n  "
            + "\n  ".join(unresolved)
            + "\nAdd them to data/processed/category_to_cameo.json "
            "(or use a CODES.csv name)."
        )

    targets = {
        resolveCategoryIndex(label, categoryToCameo, labelToIndex) for label in labels
    }
    print(f"{source}: {len(labels)} labels resolve to {len(targets)} training classes.")
    return labelToIndex
