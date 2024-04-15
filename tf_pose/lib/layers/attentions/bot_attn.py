import tensorflow as tf
from typing import Optional
from tensorflow.keras.layers import Conv2D,Dense, Dropout, Add
from lib.layers.pos_embs import RelativePositionalEmbedding

class RelativePositionMultiHeadAttention(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    R""" MHSA_RelPosEmbed (BottleneckAttention, BottlenecTransformer)


    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    x = Input(shape=(32,32,128))
    out = RelativePositionMultiHeadAttention(
        out_channels  = None,
        num_heads = 4,
        key_dims = 0,
        output_weight = False, 
        output_bias = False, 
        attn_dropout_rate  =.0,      
        name='BottleneckMHSA'
    )(x)

    model = Model(x, out)
    model.summary(150)
    model.get_layer('BottleneckMHSA').get_config()
    model.save('test')
    tensor = tf.ones(shape=(1,32,32,128))
    model(tensor)
    jit_model = tf.function(model, jit_compile=True)
    %timeit _ = jit_model(tensor)
    """
    def __init__(
        self, 
        out_channels : Optional[int] = None,
        num_heads : int = 4,
        key_dims : int = 0,
        output_weight : bool = False, 
        output_bias : bool = False, 
        attn_dropout_rate : float =.0,
        **kwargs 
    ):
        super().__init__(**kwargs)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.key_dims = key_dims
        self.output_weight = output_weight
        self.output_bias = output_bias
        self.attn_dropout_rate = attn_dropout_rate


    def build(self, input_shape):

        _, self.H, self.W, self.c  = input_shape
        if self.out_channels is None  or self.out_channels < 0 :
            self.out_channels = self.c

        if self.key_dims <= 0 :#
            self.key_dims = self.c  // self.num_heads   #Default value

        self.qk_dims = self.num_heads * self.key_dims
        #qk_dims*2 + self.out_channels,
        self.qkv_conv = Conv2D(
            (self.qk_dims*2 + self.out_channels),
            kernel_size=1,
            strides = 1,
            use_bias= False , 
            padding="valid",
            name="qkv_Conv"
        )
        self.rel_pos_embed = RelativePositionalEmbedding(
            use_absolute_pos = False,
            name="pos_emb"
        )
      
        self.qk_scale = self.add_weight(name="scale",
                shape=(1,),
                dtype=tf.dtypes.float32,
                initializer=tf.constant_initializer(
                    value = 1.0 / (float(self.key_dims) ** 0.5)
                ),
                trainable=False
        )

        if self.attn_dropout_rate > 0. :
            self.attn_dropout = Dropout(
                self.attn_dropout_rate,
                name ='dropout'
            )

        if self.output_weight :
            self.out_dense = Dense(
                self.out_channels,  
                use_bias=self.output_bias,
                name = 'output_Dense'
            )

    def call(self, x) :
        #b = tf.shape(x)[0] #batch_size
        r'''
        arg 

        return  
        
        
        '''

        qkv = self.qkv_conv(x)

        query, key, value = tf.split(
            qkv, 
            num_or_size_splits=[self.qk_dims, self.qk_dims,self.out_channels], 
            axis=-1
        )

        # query :(b,H,W,qk_dims)
        # key :(b,H,W,qk_dims)
        # value :(b,H,W, out_channels)
        query = tf.reshape(
            query, 
            [-1, self.H*self.W, self.num_heads, query.shape[-1]//self.num_heads]
        )#(b,H,W,qk_dims) => (b, H*W, num_heads, key_dim)
        query = tf.transpose( query, [0,2,1,3]) #(b,  num_heads, H*W,  key_dim)

        key = tf.reshape(
            key, 
            [-1, self.H*self.W, self.num_heads, key.shape[-1]//self.num_heads]
        )#:(b,H,W,qk_dims) => (B, H*W,num_heads,key_dim)
        key = tf.transpose( key, [0,2,3,1]) #(b,num_heads,key_dim, H*W)

        value = tf.reshape(
            value, 
            [-1, self.H*self.W, self.num_heads, value.shape[-1]//self.num_heads]
        )#(b,H,W, out_channels) => (b, H*W, num_heads, out_channels/num_heads=v_dim)

        value = tf.transpose(value, [0,2,1,3]) #(b,num_heads, H*W, v_dim)

        pos_query = tf.reshape(
            query, 
            [-1, self.num_heads, self.H, self.W, self.key_dims]
        ) #(b, num_heads, H*W,  key_dim) => (b,num_heads, H, W, key_dims)
        
        pos_emb = self.rel_pos_embed(pos_query) #(b,num_heads, H, W, H, W)


        attention_scores  = tf.matmul(
            query ,  key
        ) * self.qk_scale
        #(b,  num_heads, H*W, key_dim)@(b,num_heads,key_dim, H*W) =>(b,num_heads, H*W, H*W)
 
        pos =  tf.reshape(
            pos_emb, [-1, *attention_scores.shape[1:]]
        ) #(b,num_heads, H, W, H, W) => #(b,num_heads, H*W, H*W)


        attention_scores = Add()([attention_scores, pos]) #(b,num_heads, H*W, H*W)

        attention_scores = tf.nn.softmax(attention_scores, axis=-1)

        if hasattr( self, 'attn_dropout'):
           attention_scores = self.attn_dropout(attention_scores) 

        attention_output = tf.matmul(
            attention_scores, value
        ) # (b,num_heads, H*W, H*W) @(b,num_heads, H*W, v_dim) => (b,num_heads, H*W, v_dim)
        attention_output = tf.transpose(
            attention_output, [0, 2, 1, 3]
        )# (b,num_heads, H*W, out_channels) => (b,, H*W, num_heads, v_dim)
        attention_output = tf.reshape(
            attention_output, 
            [-1, self.H, self.W, tf.reduce_prod(attention_output.shape[1:])/(self.H*self.W)]
        ) #(b,, H*W, num_heads, v_dim) => (b,, H, W, num_heads*v_dim), 
        #note :  here, num_heads*v_dim=out_channels

        if hasattr( self, 'out_dense'):
            attention_output = self.out_dense(
                attention_output
            )
        return attention_output
        
    def get_config(self):
        config = super().get_config()
        config.update(
                {
                "out_channels"  : self.out_channels,
                "num_heads": self.num_heads,
                "key_dims" : self.key_dims ,
                "qk_dims" : self.qk_dims ,
                "attn_dropout_rate": self.attn_dropout_rate,
                "output_bias": self.output_bias,  
                }
        )
        return config 