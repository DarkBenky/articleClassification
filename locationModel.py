from transformers import AutoTokenizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model_layers import PositionalEmbedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, Embedding
import numpy as np
import os
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import json
from CountryCodes import codeToName


TEST_TEXTS = [
    "Russia declares martial law in annexed Ukrainian territories, orders mass evacuation. this unprecedented move comes as tensions escalate in the region, with Moscow citing security concerns and the need to protect its interests. The martial law declaration allows for increased military presence and restrictions on civilian activities, signaling a significant escalation in the ongoing conflict between Russia and Ukraine.",
    "The White House announced new executive orders today as Congress debates the federal budget. The President met with senior advisers at the Oval Office to discuss domestic economic policy and infrastructure spending.",
    "Flooding in Bangladesh has displaced over a million people after record monsoon rains swept through the Ganges delta. Aid organizations are struggling to reach affected villages.",
    "Tech giants in Silicon Valley reported record quarterly earnings as AI chip demand surges. Several San Francisco-based startups announced major funding rounds driven by artificial intelligence investments.",
    "The European Central Bank raised interest rates for the third consecutive time amid persistent inflation across the eurozone. Markets in Frankfurt and Paris fell sharply in response.",
    "Wildfires continue to burn across New South Wales and Victoria, forcing thousands of residents to evacuate. Australian authorities warn the fire season could be the worst in a decade.",
    "Beijing announced sweeping new regulations targeting the domestic technology sector, requiring local data storage and government oversight of algorithms used by major platforms.",
    "Israeli and Palestinian officials held indirect talks in Cairo brokered by Egyptian mediators, seeking a ceasefire agreement as violence continues in Gaza.",
    "Japan's parliament approved a new defense budget doubling military spending over the next five years, citing regional security threats and the need to modernize its Self-Defense Forces.",
    "The United Nations released a report warning that global food insecurity is worsening due to climate change, conflict, and economic instability, with millions at risk of hunger worldwide.",
]


class BestValCheckpoint(Callback):
    """Evaluates on val data every N batches, saves on best val_loss, logs test predictions to wandb."""
    def __init__(self, filepath, unique_locations, val_ds, save_freq_batches=2500, val_steps=100, warmup_batches=5000):
        super().__init__()
        self.filepath = filepath
        self.unique_locations = unique_locations
        self.location_keys = list(unique_locations.keys())
        self.save_freq_batches = save_freq_batches
        self.val_ds = val_ds
        self.val_steps = val_steps
        self.warmup_batches = warmup_batches
        self.best_val_loss = float("inf")
        self._batch_count = 0

    def _evaluate(self, ds):
        """Evaluate without disrupting the training progress bar metrics."""
        # Save running metric state so model.evaluate() doesn't reset the training accumulators
        metric_states = [[v.numpy() for v in m.variables] for m in self.model.metrics]
        val_loss, val_acc = self.model.evaluate(ds, verbose=0)
        for m, state in zip(self.model.metrics, metric_states):
            for v, s in zip(m.variables, state):
                v.assign(s)
        return val_loss, val_acc

    def _check_and_save(self, val_loss, val_acc, label):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.model.save(self.filepath)
            print(f"  -> new best val_loss ({label}) — saved to {self.filepath}")
            self._log_test_predictions(val_loss, val_acc)

    def on_train_batch_end(self, batch, logs=None):
        self._batch_count += 1
        if self._batch_count % self.save_freq_batches != 0:
            return
        val_loss, val_acc = self._evaluate(self.val_ds.take(self.val_steps))
        wandb.log({"batch_val_loss": val_loss, "batch_val_accuracy": val_acc})
        print(f"\nBatch {self._batch_count}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if self._batch_count < self.warmup_batches:
            print(f"  -> warmup ({self._batch_count}/{self.warmup_batches} batches), skipping save")
            return
        self._check_and_save(val_loss, val_acc, f"batch {self._batch_count}")

    def on_epoch_end(self, epoch, logs=None):
        # evaluate on the full val set at end of each epoch — more reliable than batch samples
        val_loss, val_acc = self._evaluate(self.val_ds)
        wandb.log({"epoch_full_val_loss": val_loss, "epoch_full_val_accuracy": val_acc, "epoch": epoch})
        print(f"\nEpoch {epoch + 1} full val: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        self._check_and_save(val_loss, val_acc, f"epoch {epoch + 1}")

    def _log_test_predictions(self, val_loss, val_acc):
        rows = []
        for text in TEST_TEXTS:
            inputs = tokenizer(text, truncation=True, padding="max_length",
                               max_length=CONTEXT_SIZE, return_tensors="np")
            preds = self.model.predict(inputs["input_ids"], verbose=0)
            top_indices = np.argsort(preds[0])[-3:][::-1]
            top_preds = " | ".join(
                f"{codeToName(self.location_keys[i])} ({preds[0][i]:.3f})" for i in top_indices
            )
            rows.append([text[:100] + "...", top_preds])
        table = wandb.Table(columns=["text", "top_3_predictions"], data=rows)
        wandb.log({"test_predictions": table, "best_val_loss": val_loss, "best_val_accuracy": val_acc})

CONTEXT_SIZE = 512
EPOCHS = 10
BATCH_SIZE = 32

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

TOKENIZED_DIR = "/media/user/2TB/tokenizedtext"


def buildModel(output_dim, vocab_size, embedding_dim=128, kernel_sizes=[3, 5, 7], conv_units=512, units=512, dropout_rate=0.2, denseLayers=1, num_heads=4, num_transformer_blocks=4):
    inputs = keras.Input(shape=(CONTEXT_SIZE,))
    x = PositionalEmbedding(vocab_size=vocab_size, context_size=CONTEXT_SIZE, embedding_dim=embedding_dim, name="positional_embedding")(inputs)

    # Stacked transformer encoder blocks (attention -> Add+Norm -> FFN -> Add+Norm)
    for _ in range(num_transformer_blocks):
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim // num_heads, dropout=dropout_rate)(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        ffn = layers.Dense(embedding_dim * 4, activation="relu")(x)
        ffn = layers.Dropout(dropout_rate)(ffn)
        ffn = layers.Dense(embedding_dim)(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization()(x)

    branches = []
    for ks in kernel_sizes:
        branch = layers.Conv1D(conv_units, ks, activation="relu", padding="same")(x)
        branch = layers.GlobalMaxPooling1D()(branch)
        branches.append(branch)

    x = layers.Concatenate()(branches) if len(branches) > 1 else branches[0]
    x = layers.Dropout(dropout_rate)(x)

    for _ in range(denseLayers):
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(output_dim, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    MODEL_PATH = "location_model_best.keras"

    with open("unique_fips_locations.json", "r") as f:
        unique_locations = json.load(f)

    if not os.path.exists(MODEL_PATH):
        with open(os.path.join(TOKENIZED_DIR, "meta.json"), "r") as f:
            meta = json.load(f)

        valid_total = meta.get("valid_total", meta["total"])
        print(f"Loading {valid_total} tokenized items from {TOKENIZED_DIR}")

        X_mmap = np.memmap(os.path.join(TOKENIZED_DIR, "X.dat"), dtype="int32", mode="r", shape=(valid_total, CONTEXT_SIZE))
        y_mmap = np.memmap(os.path.join(TOKENIZED_DIR, "y.dat"), dtype="int64", mode="r", shape=(valid_total,))

        val_size = int(valid_total * 0.03)
        train_size = valid_total - val_size

        rng = np.random.default_rng(42)
        all_indices = rng.permutation(valid_total).astype(np.int64)
        val_indices   = all_indices[:val_size]
        train_indices = all_indices[val_size:]

        def fetch_batch(batch_idx):
            x = tf.numpy_function(lambda b: X_mmap[b], [batch_idx], tf.int32)
            y = tf.numpy_function(lambda b: y_mmap[b], [batch_idx], tf.int64)
            x.set_shape([None, CONTEXT_SIZE])
            y.set_shape([None])
            return x, y

        train_ds = (tf.data.Dataset.from_tensor_slices(train_indices)
                    .shuffle(buffer_size=200_000, reshuffle_each_iteration=True)
                    .batch(BATCH_SIZE)
                    .map(fetch_batch, num_parallel_calls=tf.data.AUTOTUNE)
                    .prefetch(tf.data.AUTOTUNE))

        val_ds = (tf.data.Dataset.from_tensor_slices(val_indices)
                  .batch(BATCH_SIZE)
                  .map(fetch_batch, num_parallel_calls=tf.data.AUTOTUNE)
                  .prefetch(tf.data.AUTOTUNE))

        with wandb.init(project="article-classification") as run:
            model = buildModel(output_dim=len(unique_locations), vocab_size=tokenizer.vocab_size, embedding_dim=128, kernel_sizes=[3, 5, 7], conv_units=512, units=1024, dropout_rate=0.2, denseLayers=1, num_heads=4, num_transformer_blocks=2)
            model.build(input_shape=(None, CONTEXT_SIZE))
            model.summary()

            wandb.config.update({"model_size": model.count_params(), "train_size": train_size, "val_size": val_size})

            checkpoint_best = BestValCheckpoint(MODEL_PATH, unique_locations=unique_locations, val_ds=val_ds, save_freq_batches=2048, val_steps=32, warmup_batches=4096)

            model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[
                WandbMetricsLogger(log_freq="batch"),
                checkpoint_best,
            ])

        model.save("location_classification_model.h5")
    else:
        print(f"Found existing model at {MODEL_PATH}, skipping training.")

    print(f"\nLoading model from {MODEL_PATH} for inference...")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"PositionalEmbedding": PositionalEmbedding})

    test_texts = TEST_TEXTS

    print("\nTop predicted locations:")
    print("-" * 60)
    for text in test_texts:
        inputs = tokenizer(text, truncation=True, padding="max_length", max_length=CONTEXT_SIZE, return_tensors="np")
        predictions = model.predict(inputs["input_ids"], verbose=0)
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        print(f"TEXT: {text[:80]}...")
        for idx in top_indices:
            location = list(unique_locations.keys())[idx]  # already a FIPS code
            confidence = predictions[0][idx]
            location_name = codeToName(location)
            print(f"  {location_name}: {confidence:.4f}")
        print()
