
from tensorflow.keras import initializers
import numpy as np
import tensorflow as tf

 
############################################################################
#
#
############################################################################
class RotaryPositionEmbedding(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r"""Rotary positional encoding layer.

    This layer encodes absolute positional information with a rotation
    matrix. It calculates the rotary encoding with a mix of sine and
    cosine functions with geometrically increasing wavelengths.
    Defined and formulated in [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v4).
    The input must be a tensor with shape a sequence dimension and a feature
    dimension. Typically, this will either an input with shape
    `(batch_size, sequence_length, feature_length)` or
    `(batch_size, sequence_length, num_heads, feature_length)`.
    This layer will return a new tensor with the rotary embedding applied to
    the input tensor.

    Args:
        max_wavelength (int): The maximum angular wavelength of the sine/cosine
            curves.
        scaling_factor (float): float. The scaling factor used to scale frequency range.
            Default to 1
        sequence_axis (int):  Sequence axis in the input tensor. Default to 1
        feature_axis (int):  Feature axis in the input tensor. Default to -1
        start_index (int): An integer or integer tensor. The starting position to
            compute the rotary embedding from. This is useful during cached
            decoding, where each position is predicted separately in a loop. Default to 0


    Call arguments:
        inputs: The tensor inputs to apply the embedding to. This can have
            any shape, but must contain both a sequence and feature axis. The
            rotary embedding will be applied to `inputs` and returned.
        start_index: An integer or integer tensor. The starting position to
            compute the rotary embedding from. This is useful during cached
            decoding, where each position is predicted separately in a loop.

    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    x = Input(shape=(17, 2, 1024))
    out = RotaryPositionEmbedding(       
        max_wavelength=10000,
        scaling_factor=1.0,
        sequence_axis=1,
        feature_axis=-1,
        name ='rope'
    )(x)
    model = Model(x, out)
    model.summary(200)
    model.get_layer('rope').get_config()
    model.save(filepath='test')
    
    """

    def __init__(
        self,
        max_wavelength=10000,
        scaling_factor=1.0,
        sequence_axis=1,
        feature_axis=-1,
        start_index : int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength
        self.sequence_axis = sequence_axis
        self.feature_axis = feature_axis
        self.scaling_factor = scaling_factor
        self.start_index = start_index
        #self.built = True

    def build(self,input_shape):
        self.input_ndim = len(input_shape)
        self.seq_len = input_shape[self.sequence_axis]
        self.rotary_dim =input_shape[-1]
        self.cos_emb, self.sin_emb = self._compute_cos_sin_embedding(
            self.seq_len, self.rotary_dim, self.start_index
        )

    def call(self, inputs):
        """ 
        Call arguments:
            inputs: The tensor inputs to apply the embedding to. This can have
                any shape, but must contain both a sequence and feature axis. The
                rotary embedding will be applied to `inputs` and returned.
        """
        #rotary_dim = tf.shape(inputs)[-1]
        # cos_emb, sin_emb = self._compute_cos_sin_embedding(
        #     inputs, rotary_dim, start_index
        # )
        return self._apply_rotary_pos_emb(inputs, self.cos_emb, self.sin_emb)

    def _apply_rotary_pos_emb(self, tensor, cos_emb, sin_emb):
        x1, x2 = tf.split(tensor, 2, axis=self.feature_axis)
        half_rot_tensor = tf.concat((-x2, x1), axis=self.feature_axis)
        return (tensor * cos_emb) + (half_rot_tensor * sin_emb)

    def _compute_cos_sin_embedding(self, seq_len, rotary_dim, start_index):
        freq_range = tf.range(0, rotary_dim, 2, dtype="float32")
        freq_range = tf.cast(freq_range, self.compute_dtype)
        freq_range = freq_range / tf.cast(
            self.scaling_factor, self.compute_dtype
        )
        inverse_freq = 1.0 / (
            self.max_wavelength
            ** (freq_range / tf.cast(rotary_dim, self.compute_dtype))
        )
        #seq_len = tf.shape(x)[self.sequence_axis]
        tensor = tf.range(seq_len, dtype="float32") + start_index
        tensor = tf.cast(tensor, dtype=inverse_freq.dtype)
        freq = tf.einsum("i, j -> ij", tensor, inverse_freq)
        embedding = tf.concat((freq, freq), axis=self.feature_axis)

        def get_axis(axis):
            return axis if axis > 0 else  self.input_ndim + axis

        feature_axis = get_axis(self.feature_axis)
        sequence_axis = get_axis(self.sequence_axis)

        for axis in range( self.input_ndim):
            if axis != sequence_axis and axis != feature_axis:
                embedding = tf.expand_dims(embedding, axis)
        return tf.math.cos(embedding), tf.math.sin(embedding)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_wavelength": self.max_wavelength,
                "scaling_factor": self.scaling_factor,
                "sequence_axis": self.sequence_axis,
                "feature_axis": self.feature_axis,
                "seq_len": self.seq_len,
                "rotary_dim": self.rotary_dim,
            }
        )
        return config