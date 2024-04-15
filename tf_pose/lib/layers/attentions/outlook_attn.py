import math
import tensorflow as tf
from tensorflow.keras.layers import  Dense, Dropout, ZeroPadding2D,  AveragePooling2D
from ..functional_ops import UnFoldLayer, FoldLayer


#----------------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------------
class OutlookAttention(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    R""" OutlookAttention (OutlookAttention)


    #https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/volo/volo.py


    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    x = Input(shape=(256,192,32))
    out = OutlookAttention(
            embed_dim = 128,
            num_heads  = 8,
            kernel_size  = 3, 
            strides = 2,
            attn_dropout_rate  =.0,
            output_dropout_rate=.0,
            name='OutlookAttention'
    )(x)
    model = Model(x, out)
    model.summary(150)  

    """
    def __init__(self, 
                embed_dim : int,
                num_heads : int = 8,
                kernel_size : int = 3, 
                strides : int = 2,
                attn_dropout_rate : float =.0,
                output_dropout_rate : float =.0,
                **kwargs ):
        super().__init__(**kwargs)

        self.k = kernel_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.strides = strides
        self.attn_dropout_rate = attn_dropout_rate
        self.output_dropout_rate = output_dropout_rate


    def build(self, input_shape):

        _, self.H, self.W, self.c  = input_shape
        self.hh, self.ww = int(math.ceil( self.H / self.strides )), int(math.ceil(self.W / self.strides ))
        #self.hh, self.ww = self.H // self.strides , self.W//self.strides 

        qk_scale =  1.0 / (float(self.embed_dim // self.num_heads) ** 0.5)
        self.qk_scale = self.add_weight(name="scale",
                shape=(1,),
                dtype=tf.dtypes.float32,
                initializer=tf.constant_initializer(value=qk_scale),
                trainable=False)
        
        self.v_dense = Dense(
            self.embed_dim, use_bias=False
        )
        
        'attn'
        pool_padding = "valid" if (self.H % self.strides == 0 and self.W % self.strides == 0) else "same"
        self.attn_avg_pool = AveragePooling2D(
            pool_size=self.strides, strides=self.strides, padding=pool_padding
        )
        self.attn_dense = Dense(
            pow(self.k,4)*self.num_heads, use_bias=True
        )
        if self.attn_dropout_rate>0. :
            self.attn_dropout = Dropout(
                self.attn_dropout_rate
            )
        'unfold / fold'
        self.unfold = UnFoldLayer(
            kernel_size = self.k, 
            strides  = self.strides,
            dilation_rate  = 1,
            use_conv = False,
            compressed_output =False,
            name='UnFold'
        )
        self.fold = FoldLayer(
            output_shape = (self.H, self.W),
            kernel_size = self.k, 
            strides  = self.strides,
            dilation_rate  = 1,
            compressed_input = False,
            name='Fold'
        )
        'output'
        self.out_dense = Dense(
           self.embed_dim, use_bias=True
        )
        if self.output_dropout_rate >0. :
            self.out_dropout = Dropout(
                self.out_dropout_rate
            )

    def call(self, x) :

        v = self.v_dense(x) #(b,H,W,embed_dim)
        "attntion"
        attn = self.attn_avg_pool(x) #(b,h,w,c)
        attn = self.attn_dense(attn) #(b,h,w, num_head*k**4)
        attn = attn*self.qk_scale #(b,h,w, num_head*k**4)
        attn = tf.reshape(
            attn, [-1,self.hh,self.ww, self.num_heads, pow(self.k,2), pow(self.k,2)]
        ) #(b,H, W, num_head, k**2, k**2)
        attn = tf.nn.softmax(attn , axis=-1)
        if hasattr(self,'attn_dropout'):
           output = self.attn_dropout(output)

        """ unfold to extract patches"""
        patches = self.unfold(v) #(b,h, w, k,k, embed_dim)
        
        mm = tf.reshape(
            patches, [-1,self.hh,self.ww, pow(self.k,2), self.num_heads, self.embed_dim//self.num_heads]
        ) #(b,h, w, k**2, num_head, embed_dim//num_head)
        mm = tf.transpose(mm,[0, 1, 2, 4, 3, 5]) #(b,h, w, num_head,  k**2, embed_dim//num_head)
        
        mm = tf.matmul(attn, mm)  #(b,h, w, num_head,  k**2, embed_dim//num_head)
        mm = tf.transpose(mm,[0, 1, 2, 4, 3, 5]) #(b,h, w, k**2, num_head,  embed_dim//num_head)
        mm = tf.reshape(mm, [-1, self.hh, self.ww, self.k, self.k, self.embed_dim]) #(b,h,w,k,k,embed_dim)

        """ fold """
        output = self.fold(mm) # #(b,h,w,k,k,embed_dim) => (b,H,W,embed_dim)  
        output = self.out_dense(output)
        
        if hasattr(self,'out_dropout'):
           output = self.out_dropout(output)

        return output
    
    def get_config(self):
        config = super().get_config()
        config.update(
                {
                'kernel_size' : self.k,
                'embed_dim' : self.embed_dim,   
                "num_heads": self.num_heads,
                "strides": self.strides,
                "attn_dropout_rate": self.attn_dropout_rate,
                "output_dropout_rate": self.output_dropout_rate,
                }
        )
        return config 
    

#----------------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------------
class SimpleOutlookAttention(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    R""" OutlookAttention (OutlookAttention)


    #https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/volo/volo.py

    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    #print(out.shape)
    x = Input(shape=(256,192,32))
    out = SimpleOutlookAttention(
            embed_dim = 128,
            num_heads  = 8,
            kernel_size  = 3, 
            attn_dropout_rate  =.0,
            name='OutlookAttention'
    )(x)
    model = Model(x, out)
    model.summary(150)
    model.get_layer('OutlookAttention').get_config()
    model.save("test")
    jit_model = tf.function(model, jit_compile=True)
    tensor = tf.ones(shape=(1,256,192,32))
    %timeit _ = jit_model(tensor)

    
    """
    def __init__(self, 
                embed_dim : int,
                num_heads : int = 8,
                kernel_size : int = 3, 
                attn_dropout_rate : float =.0,
                **kwargs ):
        super().__init__(**kwargs)

        self.k = kernel_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout_rate = attn_dropout_rate


    def build(self, input_shape):

        _, self.H, self.W, self.c  = input_shape
        self.hh, self.ww = int(math.ceil(self.H /self.k )), int(math.ceil(self.W/self.k ))
        self.key_dim = self.embed_dim //self.num_heads

        self.padded_h = self.hh*self.k - self.H
        self.padded_w = self.ww*self.k - self.W
        if self.padded_h != 0 or self.padded_w != 0 :
            self.zeros_padding = ZeroPadding2D(((0, self.padded_h), (0, self.padded_w)))

        qk_scale =  1.0 / (float(self.embed_dim // self.num_heads) ** 0.5)
        self.qk_scale = self.add_weight(name="scale",
                shape=None,
                dtype=tf.dtypes.float32,
                initializer=tf.constant_initializer(value=qk_scale),
                trainable=False)
        
        self.v_dense = Dense(
            self.embed_dim, use_bias=False
        )
        
        'attn'
        self.attn_avg_pool = AveragePooling2D(
            pool_size=self.k, strides=self.k,
        )
        self.attn_dense = Dense(
            pow(self.k,4)*self.num_heads, use_bias=True
        )
        if self.attn_dropout_rate>0. :
            self.attn_dropout = Dropout(
                self.attn_dropout_rate
            )
        'output'
        self.out_dense = Dense(
           self.embed_dim, use_bias=True
        )

    def call(self, x) :
        if  hasattr(self, 'zeros_padding'): 
            x = self.zeros_padding(x) #(b,H,W,c)


        v = self.v_dense(x) #(b,H,W,embed_dim)
        v = tf.reshape(
            v, [-1,self.hh,self.k, self.ww, self.k, self.num_heads, self.key_dim]
        ) #(b,H, W, num_head, k**2, k**2)
        v = tf.transpose(v, [0, 1, 3, 5, 2, 4, 6])
        v = tf.reshape(v, [-1, self.hh, self.ww, self.num_heads, pow(self.k,2), self.key_dim])


        "attntion"
        attn = self.attn_avg_pool(x) #(b,h,w,c)
        attn = self.attn_dense(attn) #(b,h,w, num_head*k**4)
        attn = attn*self.qk_scale #(b,h,w, num_head*k**4)
        attn = tf.reshape(
            attn, [-1,self.hh,self.ww, self.num_heads, pow(self.k,2), pow(self.k,2)]
        ) #(b,H, W, num_head, k**2, k**2)
        attn = tf.nn.softmax(attn , axis=-1)
        if hasattr(self,'attn_dropout'):
           output = self.attn_dropout(output)

        out = tf.matmul(attn, v)  #(b,h, w, num_head,  k**2, embed_dim//num_head)
        out = tf.reshape(
            out, [-1,self.hh,self.ww, self.num_heads, self.k, self.k, self.key_dim]
        ) #(b,h, w, k**2, num_head, embed_dim//num_head)
        out = tf.transpose(
            out, [0, 1, 4, 2, 5, 3, 6]
        ) #(b,h, w, k**2, num_head,  embed_dim//num_head)
        out = tf.reshape(
            out, [-1, self.H+self.padded_h , self.W+self.padded_w, self.embed_dim]
        )  # [1, 28, 28, 192]

        out = tf.slice(
            out,
            begin = [0,0,0,0],
            size =  [-1, self.H,self.W,-1]
        )
        out = self.out_dense(out)
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update(
                {
                'kernel_size' : self.k,
                'embed_dim' : self.embed_dim,   
                "num_heads": self.num_heads,
                "attn_dropout_rate": self.attn_dropout_rate,
                }
        )
        return config 
    
