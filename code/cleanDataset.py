import ast
import random
from itertools import islice

import os
import sys
import hashlib
import pickle

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")),
)
from project_config import (
    CORPUS_PATH, CLEANED_CORPUS_PATH,ARTICLE_HASHES_PATH
)

CHUNK_SIZE = 8196

def readChunks(chunkSize):
    with open(CORPUS_PATH, encoding="utf-8") as f:
        while True:
            lines = list(islice(f, chunkSize))
            if not lines:
                return
            records = [ast.literal_eval(line) for line in lines]
            random.shuffle(records)
            yield records

# remove special characters, remove html tags, remove redacted whitespace
# ask decision model if relevant
# q = is this text high quality news article? 
# x = is this category correct?
# cat = select correct category
# if q < 0.7:
#    skip
# if x < 0.7:
#    use category from model
# save to cleaned corpus
# save hash of this article to avoid duplicates
def processChunk(chunk):