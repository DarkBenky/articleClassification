import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    """Combines token and learned positional embeddings into a single serializable layer."""
    def __init__(self, vocab_size, context_size, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="token_embedding")
        self.pos_emb   = layers.Embedding(input_dim=context_size, output_dim=embedding_dim, name="position_embedding")

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        return self.token_emb(x) + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"vocab_size": self.vocab_size, "context_size": self.context_size, "embedding_dim": self.embedding_dim})
        return config
