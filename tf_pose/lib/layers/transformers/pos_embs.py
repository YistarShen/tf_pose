import tensorflow as tf

############################################################################
#
#
############################################################################
class AddPositionEmbedding(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""
    def __init__(self, **kwargs ):
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert (len(input_shape) == 3), \
        f"Number of dimensions should be 3, got {len(input_shape)}"

        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
