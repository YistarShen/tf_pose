import tensorflow as tf
from typing import Optional
from ..functional_ops  import UnFoldLayer
from ..transformers import RelativePositionalEmbedding
from tensorflow.keras.layers import Conv2D,Dense, Dropout, Add, AveragePooling2D

class HaloAttention(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    R""" HaloAttention (HaloAttention)


    #https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/volo/volo.py

    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    #print(out.shape)
    x = Input(shape=(256,192,32))
    out = HaloAttention(
        out_channels = 256,
        num_heads  = 8,
        block_size = 4,
        kernel_size  = 1, 
        strides = 2,
        output_weight =True, 
        output_bias = False,        
        name='HaloAttention'
    )(x)

    model = Model(x, out)
    model.summary(150)
    model.get_layer('HaloAttention').get_config()
    model.save('test')
    jit_model = tf.function(model, jit_compile=True)
    tensor = tf.ones(shape=(1,256,192,32))
    %timeit _ = jit_model(tensor)

    """
    def __init__(self, 
                out_channels : Optional[int] = None,
                num_heads : int = 8,
                block_size : int = 4,
                kernel_size : int = 1, 
                strides : int = 1,
                output_weight : bool =True, 
                output_bias : bool = False, 
                attn_dropout_rate : float =.0,
                **kwargs ):
        super().__init__(**kwargs)

        self.out_channels = out_channels
        self.k = kernel_size
        self.num_heads = num_heads
        self.strides = strides
        self.block_size = block_size
        self.output_weight = output_weight
        self.output_bias = output_bias

        self.attn_dropout_rate = attn_dropout_rate


    def build(self, input_shape):

        _, self.H, self.W, self.c  = input_shape
        if self.out_channels is None  or self.out_channels < 0 :
            self.out_channels = self.c

 
        self.key_dim = self.c  // self.num_heads  # Default value
        self.emb_dim = self.num_heads * self.key_dim
        self.kv_kernel = self.block_size + self.k* 2
        if self.block_size % self.strides != 0:
            self.strides = 1
            self.avg_pool_down = True
        else:
            self.avg_pool_down = False

        self.query_block = self.block_size // self.strides

        qk_scale =  1.0 / (float(self.emb_dim // self.num_heads) ** 0.5)
        self.qk_scale = self.add_weight(name="scale",
                shape=(1,),
                dtype=tf.dtypes.float32,
                initializer=tf.constant_initializer(value=qk_scale),
                trainable=False)
        

        self.q_conv = Conv2D(
            self.emb_dim,
            kernel_size=1,
            strides = self.strides,
            use_bias= False , 
            padding="valid",
            name="query_Conv"
        )
        self.hh_qq = (self.H//self.strides)//self.query_block
        self.ww_qq = (self.W//self.strides)//self.query_block
        self.cc_qq = self.emb_dim // self.num_heads

        self.kv_dim =  (self.emb_dim + self.out_channels)
        self.kv_conv = Conv2D(
            self.kv_dim,
            kernel_size=1,
            strides = 1,
            use_bias= False , 
            padding="valid",
            name="key_value_Conv"
        )
        self.unfold = UnFoldLayer(
            kernel_size = self.kv_kernel, 
            strides  = self.block_size,
            dilation_rate  = 1,
            use_conv = False,
            compressed_output =True,
            padding = 'valid',
            name='UnFold'
        )
        self.kv_hh, self.kv_ww = self.H//self.block_size, self.W//self.block_size


        self.rel_pos_embed = RelativePositionalEmbedding(
            position_height=self.kv_kernel
        )

        if self.output_weight :
            self.out_dense = Dense(
                self.out_channels, use_bias=self.output_bias
            )

        if self.attn_dropout_rate > 0. :
            self.attn_dropout = Dropout(
                self.attn_dropout_rate
            )

        if self.avg_pool_down :
            self.avg_pool = AveragePooling2D(2, strides=2)

    def call(self, x) :
        #b = tf.shape(x)[0] #batch_size

        query = self.q_conv(x) #(b,H,W,c)
        query = tf.reshape(
            query, [-1, self.hh_qq, self.query_block, self.ww_qq, self.query_block, self.num_heads, self.cc_qq]
        )
        query = tf.transpose(query, [0, 5, 1, 3, 2, 4, 6])  # [batch, num_heads, hh, ww, query_block, query_block, key_dim]
        # attn_query = [batch, num_heads, hh, ww, query_block * query_block, key_dim]
        attn_query = tf.reshape(
            query, [-1, self.num_heads, self.hh_qq, self.ww_qq, pow(self.query_block,2), self.cc_qq]
        ) * self.qk_scale  # qk_scale NOT multiplied with pos_query
        # pos_query = [batch, num_heads * hh * ww, query_block, query_block, key_dim]
        pos_query = tf.reshape(
            query, [-1, self.num_heads * self.hh_qq * self.ww_qq, self.query_block,self.query_block, self.cc_qq]
        )

        key_value  = self.kv_conv(x) #(b,H,W,  emb_dim+filters=kv_c)
        kv_padded = tf.pad(
            key_value, [[0, 0], [self.k, self.k], [self.k, self.k], [0, 0]]
        )
 
        kv_inp = self.unfold(kv_padded) #(b,hh,ww, cc)

        _, hh_kk, ww_kk, cc = kv_inp.shape
        cc_kk = cc // self.num_heads // self.kv_kernel // self.kv_kernel
        kv_inp = tf.reshape(
            kv_inp, [-1, hh_kk, ww_kk, self.kv_kernel, self.kv_kernel, self.num_heads, cc_kk]
        )
        # kv_inp = tf.reshape(
        #     kv_inp, [-1, self.kv_hh, self.kv_ww, self.kv_kernel, self.kv_kernel, self.num_heads,  self.kv_dim//self.num_heads//pow(self.kv_kernel,2)]
        # )
        kv_inp = tf.transpose(kv_inp, [0, 5, 1, 2, 3, 4, 6])
        kv_inp = tf.reshape(kv_inp, [-1, self.num_heads, hh_kk, ww_kk, pow(self.kv_kernel,2), cc_kk])
        key, value = tf.split(kv_inp, [self.emb_dim // self.num_heads, self.out_channels //self.num_heads], axis=-1)
        
        attention_scores = tf.matmul(attn_query, tf.transpose(key,[0, 1, 2, 3, 5, 4]))  #(b,h, w, num_head,  k**2, embed_dim//num_head)

        pos = self.rel_pos_embed(pos_query)
        pos = tf.reshape(pos, [-1, *attention_scores.shape[1:]])

        attention_scores = Add()([attention_scores, pos])
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        if hasattr( self, 'attn_dropout'):
           attention_scores = self.attn_dropout(attention_scores) 

        attention_output = tf.matmul(attention_scores, value)
        _, heads, hh_aa, ww_aa, patch, cc_aa = attention_output.shape
        attention_output = tf.reshape(attention_output, [-1, heads, hh_aa, ww_aa, self.query_block, self.query_block, cc_aa])
        attention_output = tf.transpose(attention_output, [0, 2, 4, 3, 5, 1, 6])
        attention_output = tf.reshape(attention_output, [-1, hh_aa*self.query_block, ww_aa*self.query_block, heads*cc_aa])

        if self.avg_pool_down :
            attention_output = self.avg_pool(attention_output)
            
        if self.output_weight:
            attention_output = self.out_dense(attention_output)

        return attention_output
    
    def get_config(self):
        config = super().get_config()
        config.update(
                {
                "out_channels"  : self.out_channels,
                'kernel_size' : self.k, 
                "num_heads": self.num_heads,
                "strides": self.strides,
                "attn_dropout_rate": self.attn_dropout_rate,
                "output_bias": self.output_bias,  
                "output_weight": self.output_weight,   
                "avg_pool_downsample": self.avg_pool_down,  
                }
        )
        return config 