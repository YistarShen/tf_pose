from tensorflow.keras import initializers
import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
class RelativePositionalEmbedding(tf.keras.layers.Layer):
    VERSION = '1.1.0'
    r"""  RelativePositionalEmbedding2D


    Examples:
    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    x = Input(shape=(8, 32,32,64))
    out = RelativePositionalEmbedding(
            position_height=6,
            name='RelativePositionalEmbedding'
    )(x)
    model = Model(x, out)
    model.summary(150)   
    model.get_layer('RelativePositionalEmbedding').show_pos_emb()
    for layer in model.layers:
        pass
    
    """
    def __init__(self, 
                 position_height=0, 
                 position_width=0, 
                 use_absolute_pos=False, 
                 dynamic_shape=False, 
                 **kwargs):
        super().__init__(**kwargs)

        self.position_height = position_height
        self.position_width = position_width if position_width > 0 else position_height
        self.use_absolute_pos = use_absolute_pos
        self.dynamic_shape = dynamic_shape  # Deprecated

    def build(self, input_shape):

        _, self.num_heads, self.input_height, self.input_width, self.key_dim = input_shape
        

        self.position_height =  max(self.position_height, self.input_height)
        self.position_width =  max(self.position_width, self.input_width)
        #self.key_dim = key_dim

        if self.use_absolute_pos:
            self.pos_emb_h_shape = (self.key_dim, self.position_height)
            self.pos_emb_w_shape = (self.key_dim, self.position_width)
        else:
            self.pos_emb_h_shape = (self.key_dim, 2 * self.position_height - 1)
            self.pos_emb_w_shape = (self.key_dim, 2 * self.position_width - 1)

        stddev = self.key_dim**-0.5
        self.pos_emb_h = self.add_weight(
            name="r_height", 
            shape=self.pos_emb_h_shape, 
            initializer=initializers.random_normal(stddev=stddev), 
            trainable=True
        )
        self.pos_emb_w = self.add_weight(
            name="r_width", 
            shape=self.pos_emb_w_shape, 
            initializer=initializers.random_normal(stddev=stddev), 
            trainable=True
        )
        #super().build(input_shape)

    def get_config(self):
        base_config = super(RelativePositionalEmbedding, self).get_config()
        base_config.update(
            {
                "position_height": self.position_height,
                "position_width": self.position_width,
                "use_absolute_pos": self.use_absolute_pos,
                "dynamic_shape": self.dynamic_shape,

            }
        )
        return base_config

    def rel_to_abs(self, rel_pos, is_height):
        """
        Converts relative indexing to absolute.
        Input: [bs+heads, height, width, 2 * pos_dim - 1]
        Output: [bs+heads, height, width, pos_dim]
        """
        # pos_dim = self.position_height if is_height else self.position_width  # Use static values
        # num_blocks = self.input_height if is_height else self.input_width  # Use static values

        # # pos_dim = (dim + 1) // 2
        # if pos_dim == 1:
        #     return rel_pos
        # if num_blocks == 1:
        #     return rel_pos[:, :, :, -pos_dim:]
        # _, hh, ww, dim = rel_pos.shape  # [bs+heads, height, width, 2 * width - 1]
        # full_rank_gap = pos_dim - num_blocks
        # # [bs+heads, height, width * (2 * pos_dim - 1)] --> [bs+heads, height, width * (2 * pos_dim - 1) - width]
        # flat_x = tf.reshape(
        #     rel_pos, [-1, hh, ww * dim]
        # )[:, :, ww - 1 : -1]
        # # [bs+heads, height, width, 2 * (pos_dim - 1)] --> [bs+heads, height, width, pos_dim]
        # # print(f">>>> {full_rank_gap = }, {flat_x.shape = }")
        # final_x = tf.reshape(
        #     flat_x, [-1, hh, ww, 2 * (pos_dim - 1)]
        # )[:, :, :, full_rank_gap : pos_dim + full_rank_gap]
        # #tf.print("final_x : ", final_x[0,0,0,:10])

        if is_height:
            pos_dim = self.position_height 
            num_blocks = self.input_height
            hh, ww, pos_emb_dim =  self.input_width, self.input_height,  self.pos_emb_h_shape[-1]
            full_rank_gap = self.position_height - self.input_height
        else:
            pos_dim = self.position_width 
            num_blocks = self.input_width
            hh, ww, pos_emb_dim = self.input_height, self.input_width,  self.pos_emb_w_shape[-1]
            full_rank_gap = self.position_width - self.input_width

        if pos_dim == 1:
            return rel_pos
        if num_blocks == 1:
            return rel_pos[:, :, :, -pos_dim:]

        flat_x = tf.reshape(
            rel_pos, [-1, hh, ww * pos_emb_dim]
        )[:, :, ww - 1 : -1]

        final_x = tf.reshape(
            flat_x, [-1, hh, ww, 2 * (pos_dim - 1)]
        )[:, :, :, full_rank_gap : pos_dim + full_rank_gap]

        #tf.print("final_x_test : ", final_x_test[0,0,0,:10])
        return final_x

    def relative_logits(self, inputs):
        # bs, heads, hh, ww, cc = inputs.shape  # e.g.: [1, 4, 14, 16, 128]
        inputs = tf.reshape(
            inputs,  
            [-1, self.input_height, self.input_width, self.key_dim]
        )  # Merge bs and heads, for supporting TFLite conversion

        rel_logits_w = tf.matmul(
            inputs, self.pos_emb_w
        )  # [4, 14, 16, 31], 2 * 16 - 1 == 31
        rel_logits_w = self.rel_to_abs(
            rel_logits_w,  is_height=False
        )  # [4, 14, 16, 16]

        query_h = tf.transpose(
            inputs, [0, 2, 1, 3]
        )  # [4, 16, 14, 128], [bs+heads, ww, hh, dims], Exchange `ww` and `hh`
        rel_logits_h = tf.matmul(
            query_h, self.pos_emb_h
        )  # [4, 16, 14, 27], 2 * 14 - 1 == 27
        rel_logits_h = self.rel_to_abs(
            rel_logits_h, is_height=True
        )  # [4, 16, 14, 14]
        rel_logits_h = tf.transpose(
            rel_logits_h, [0, 2, 1, 3]
        )  # [4, 14, 16, 14], transpose back

        logits = tf.expand_dims(rel_logits_w, axis=-2) + tf.expand_dims(rel_logits_h, axis=-1)  # [4, 14, 16, 14, 16]

        logits = tf.reshape(
            logits, 
            [ -1, self.num_heads, self.input_height, self.input_width, self.position_height, self.position_width]
        )

        return logits

    def absolute_logits(self, inputs):
        # pos_emb = tf.expand_dims(self.pos_emb_w, -2) + tf.expand_dims(self.pos_emb_h, -1)
        # return tf.einsum("bxyhd,dpq->bhxypq", inputs, pos_emb)
        rel_logits_w = tf.matmul(
            inputs, self.pos_emb_w
        )
        rel_logits_h = tf.matmul(
            inputs, self.pos_emb_h
        )
        return tf.expand_dims(rel_logits_w, axis=-2) + tf.expand_dims(rel_logits_h, axis=-1)

    def call(self, inputs):
        pos_emb = self.absolute_logits(inputs) if self.use_absolute_pos else self.relative_logits(inputs)
        if self.dynamic_shape:
            _, _, hh, ww, _ = inputs.shape
            if hh < self.position_height or ww < self.position_width:
                pos_emb = pos_emb[:, :, :, :, :hh, :ww]
        return pos_emb

    # def load_resized_weights(self, source_layer, method="bilinear"):
    #     # For input 224 --> [128, 27], convert to 480 --> [128, 30]
    #     if isinstance(source_layer, dict):
    #         source_pos_emb_h, source_pos_emb_w = list(source_layer.values())
    #     else:
    #         source_pos_emb_h, source_pos_emb_w = source_layer.pos_emb_h, source_layer.pos_emb_w  # layer
    #     source_pos_emb_h = np.array(source_pos_emb_h.detach() if hasattr(source_pos_emb_h, "detach") else source_pos_emb_h).astype("float32")
    #     source_pos_emb_w = np.array(source_pos_emb_w.detach() if hasattr(source_pos_emb_w, "detach") else source_pos_emb_w).astype("float32")

    #     image_like_h = np.expand_dims(np.transpose(source_pos_emb_h, [1, 0]), 0)
    #     resize_h = backend.numpy_image_resize(image_like_h, target_shape=(1, self.pos_emb_h.shape[1]), method=method)[0]
    #     resize_h = np.transpose(resize_h, [1, 0])

    #     image_like_w = np.expand_dims(np.transpose(source_pos_emb_w, [1, 0]), 0)
    #     resize_w = backend.numpy_image_resize(image_like_w, target_shape=(1, self.pos_emb_w.shape[1]), method=method)[0]
    #     resize_w = np.transpose(resize_w, [1, 0])

    #     self.set_weights([resize_h, resize_w])

    def show_pos_emb(self, base_size=4):
        import matplotlib.pyplot as plt

        pos_emb_h = self.pos_emb_h.detach() if hasattr(self.pos_emb_h, "detach") else self.pos_emb_h
        pos_emb_h = self.pos_emb_h.numpy() if hasattr(self.pos_emb_h, "numpy") else np.array(self.pos_emb_h)
        pos_emb_w = self.pos_emb_w.detach() if hasattr(self.pos_emb_w, "detach") else self.pos_emb_w
        pos_emb_w = self.pos_emb_w.numpy() if hasattr(self.pos_emb_w, "numpy") else np.array(self.pos_emb_w)

        #print(pos_emb_h[0,64], pos_emb_h[0,63], pos_emb_h[0,65])
        fig, axes = plt.subplots(1, 3, figsize=(base_size * 3, base_size * 1))
        axes[0].imshow(pos_emb_h)
        axes[1].imshow(pos_emb_w)
        hh_sum = np.ones([1, pos_emb_h.shape[0]]) @ pos_emb_h
        ww_sum = np.ones([1, pos_emb_w.shape[0]]) @ pos_emb_w
        axes[2].imshow(np.transpose(hh_sum) + ww_sum)
        titles = ["pos_emb_h", "pos_emb_w", "sum"]
        for ax, title in zip(axes.flatten(), titles):
            ax.set_title(title)
            ax.set_axis_off()
        fig.tight_layout()
        return fig

#----------------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------------
class AddRelativePositionBiasT5(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r"""Relative position embedding via per-head bias in T5 style.

    copy and modify from : https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/RelativePositionBias

    Reference implementation in MeshTF:
    https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L1000
    This layer implements the relative position bias used in "Exploring the Limits
    of Transfer Learning with a Unified Text-to-Text Transformer"
    (https://arxiv.org/abs/1910.10683)


    References:
        - [Reference implementation in tensorflow-nlp-layers] 
           (https://github.com/tensorflow/models/blob/v2.14.2/official/nlp/modeling/layers/position_embedding.py#L177)        
        - [Based on implementation of 'RelativePositionBias' in gau_tensorflow.py) @brandnewchoppa] 
           (https://github.com/brandnewchoppa/gau-tensorflow/blob/main/gau_tensorflow/gau_tensorflow.py)
        - [Inspired by 'T5RelativePositionBias' in layers.py  @bert4keras] 
           (https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L195)

    Args:
        scaling_factor (float):  Default to 1
        relative_attention_num_buckets (int):  Sequence axis in the input tensor. Default to 1
        relative_attention_max_distance (int):  Feature axis in the input tensor. Default to -1
        bidirectional (bool):  Default to 0


    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    x = Input(shape=( 5, 5))
    out = AddRelativePositionBiasT5(    
        name ='RelativePositionBias'
    )(x)
    model = Model(x, out)
    model.summary(200)
    model.get_layer('RelativePositionBias').get_config()
    model.save(filepath='test')

    
    """

    def __init__(
        self,
        scaling_factor : float = 1.,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        bidirectional: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.scale = scaling_factor
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.bidirectional = bidirectional
        self.relative_attention_max_distance = relative_attention_max_distance

    def _relative_position_bucket(
                                self,
                                relative_position,
                                bidirectional=True,
                                num_buckets=32,
                                max_distance=128
        ):
        """Translate relative position to a bucket number for relative attention.

        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.

        If `bidirectional=False`, then positive relative positions are invalid.

        We use smaller buckets for small absolute relative_position and larger
        buckets for larger absolute relative_positions.

        All relative positions >=max_distance map to the same bucket.

        All relative positions <=-max_distance map to the same bucket.

        This should allow for more graceful generalization to longer sequences
        than the model has been trained on.

        Args:
            relative_position: An int32 Tensor
            bidirectional: A boolean - whether the attention is bidirectional
            num_buckets: An integer
            max_distance: An integer

        Returns:
            A Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += tf.cast(tf.math.less(n, 0), tf.int32) * num_buckets
            n = tf.math.abs(n)
        else:
            n = tf.math.maximum(n, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = tf.math.less(n, max_exact)
        val_if_large = max_exact + tf.dtypes.cast(
            tf.math.log(tf.cast(n, tf.float32) / max_exact) /
            tf.math.log(max_distance / max_exact) * (num_buckets - max_exact),
            tf.int32,
        )
        val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
        ret += tf.where(is_small, n, val_if_large)
        return ret
    
    def build(self,input_shape):
        """ 
        input_shape : (b, num_head, seq_q , seq_k) or (b, seq_q , seq_k) 
        """
        self.ndims = len(input_shape)
        if not (self.ndims==4 or self.ndims==3):
            raise ValueError(
                "input_shape's ndim must be 4(b, num_head, seq_q , seq_k)  or 3(b, seq_q , seq_k)"
                f"but got shape : {input_shape} @{self.__class__.__name__}"
            )
        
        self.qlen ,self.klen = input_shape[-2:]
        self.num_heads = 1 if self.ndims==3 else input_shape[1]

        self._relative_attention_bias = self.add_weight(
                "rel_embedding",
                shape=[self.relative_attention_num_buckets, self.num_heads],
                initializer=tf.keras.initializers.TruncatedNormal(stddev=1.0),
                dtype=self.dtype,
                trainable=True
            )
 

    def call(self, inputs):
        """ 
        Call arguments:
            inputs:  qk query@key ,  shape is (batch, qlen, klen)  or (batch, num_heads, qlen, klen) 
        Returns:
            A tensor in shape of [batch, qlen, klen]  or [batch, num_heads, qlen, klen].
        """
        #batch_size = tf.shape(qk)[0]
        context_position = tf.range(self.qlen)[:, None]
        memory_position = tf.range(self.klen)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance
        )

        values = tf.nn.embedding_lookup(
            self._relative_attention_bias, rp_bucket
        ) #(qlen, klen, num_heads=1) 

        #shape (1, qlen, klen, num_heads=1) => (1, qlen, klen, num_heads=1)
        values = tf.expand_dims(
            tf.transpose(values, [2, 0, 1]),
            axis=0
        )# shape (1, num_heads, qlen, klen)
        
        if self.ndims==3:
            values = tf.squeeze(values, axis=1) 
        # values = tf.tile(
        #     values, tf.cast([batch_size,  1, 1, 1], dtype=tf.int32)
        # )*self.scale
        # shape (1, num_heads, qlen, klen) or (1, qlen, klen) , where 1 means expand_dim for batch 
        return  inputs + values*self.scale
   
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads" : self.num_heads,
                "scaling_factor": self.scale,
                "relative_attention_num_buckets" : self.relative_attention_num_buckets,
                "relative_attention_max_distance": self.relative_attention_max_distance,
                "bidirectional" : self.bidirectional,
                "input_ndims": self.ndims,     
            }
        )
        return config
    

