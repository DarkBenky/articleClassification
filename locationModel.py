from transformers import AutoTokenizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, Embedding
import ast
import random
import wandb
from wandb.integration.keras import WandbMetricsLogger
import json

CONTEXT_SIZE = 512
EPOCHS = 10
BATCH_SIZE = 32

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def loadData(count=-1):
    # load preprocessed data from txt file
    data = []
    with open("/media/user/2TB/preprocessed_data.txt", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if count != -1 and i >= count:
                break
            data.append(ast.literal_eval(line))
            if i % 100 == 0:
                print(f"Loaded {i}/{len(lines)} data points")
    return data

def tokenizeText(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=CONTEXT_SIZE, return_tensors="tf")


def buildModel(output_dim, vocab_size, embedding_dim=128, layers=2, units=128, dropout_rate=0.2, denseLayers=2):
    # simple embedding + conv1d + global max pooling + dense model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=CONTEXT_SIZE))

    for i in range(layers):
        model.add(Conv1D(units, 5, activation="relu"))
        if i == layers - 1:
            model.add(GlobalMaxPooling1D())
        model.add(Dropout(dropout_rate))
    
    for _ in range(denseLayers):
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(dropout_rate))

    model.add(Dense(output_dim, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    # load all locations
    with open("unique_locations.json", "r") as f:
        unique_locations = json.load(f)

    location_to_idx = {loc: idx for idx, loc in enumerate(unique_locations.keys())}
    idx_to_location = {idx: loc for loc, idx in location_to_idx.items()}

    data = loadData(1_000_000)  # load first 1 million data points for testing

    wandb.init(project="article-classification", name="location-classification")

    print(f"Loaded {len(data)} data points")

    # Build model once so weights persist across epochs
    model = buildModel(output_dim=len(unique_locations), vocab_size=tokenizer.vocab_size, embedding_dim=378, layers=3, units=1024, dropout_rate=0.2, denseLayers=3)
    model.build(input_shape=(None, CONTEXT_SIZE))
    model.summary()

    # Filter dataset once: drop None locations, subsample 'Unknown' to reduce noise
    filtered_data = []
    count = 0
    l = len(data)
    for item in data:
        if item['location'] is None:
            continue
        if item['location'] == 'Unknown' and random.random() < 0.985:
            continue
        filtered_data.append(item)
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count}/{l} data points")


    print(f"Filtered to {len(filtered_data)} data points")

    X = []
    y = []
    count = 0
    for item in filtered_data:
        tokenizedText = tokenizeText(item["text"])
        X.append(tokenizedText["input_ids"][0])
        y.append(location_to_idx[item["location"]])
        count += 1
        if count % 1000 == 0:
            print(f"Tokenized {count}/{len(filtered_data)} data points")

    del filtered_data  # free memory
    del data  # free memory

    X = tf.convert_to_tensor(X)
    y = tf.convert_to_tensor(y)

    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=[WandbMetricsLogger()])

    wandb.finish()

    # save model
    model.save("location_classification_model.h5")