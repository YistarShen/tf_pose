

from tensorflow.keras.layers import  Dense, Dropout, LayerNormalization
import tensorflow as tf
#from lib.layers import AddRelativePositionBiasT5, RotaryPositionEmbedding
from .rel_pos_embs import AddRelativePositionBiasT5
from .rope import RotaryPositionEmbedding
from lib.Registers import LAYERS



#----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
class OffsetScale(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r""" OffsetScale 

    Apply per-dim scalars (gamma) and offsets (beta) to 'x' (similar to the
    learnable variables in LayerNorm).
    Used in Gated Attention Unit to create Q, K as two cheap transformations
    to Z.

    References:
        - [Paper : Offset Scale (OffsetScale)] 
           (https://arxiv.org/pdf/2202.10447.pdf)
        - [Based on implementation of 'OffsetScale' in FLASH-pytorch.py @lucidrains] 
           (https://github.com/lucidrains/FLASH-pytorch)
        - [Based on implementation of 'OffsetScale' in gau_tensorflow.py) @brandnewchoppa] 
           (https://github.com/brandnewchoppa/gau-tensorflow/blob/main/gau_tensorflow/gau_tensorflow.py)
        - [Inspired by 'ScaleOffset' in layers.py  @bert4keras] 
           (https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L195)
    Args:
        heads (int): out_channels, defaults to -1 that means inplanes=in_channels=out_channels
        split_heads (float) : epsilon of layer normalization , defaults to 1e-5.


    Note:
        - https://arxiv.org/pdf/1706.03762.pdf (based on query, key)

    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    x = Input(shape=(64,48,128))
    out = PolarizedSelfAttention(
            inplanes = -1,
            mode ="p",                
            ln_epsilon = 1e-5,
            name = 'PSAp',
            dtype= tf.float32
    )(x)
    model = Model(x, out)
    model.summary(200)
    """
    def __init__(self,
                heads : int = 2,
                split_heads : bool =True,
                **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.split_heads = split_heads

    def build(self, input_shape):

        d = input_shape[-1]
        self.beta = self.add_weight(
            name = 'beta',
            shape = (self.heads, d),
            initializer = tf.keras.initializers.zeros()
        )
            
        self.gamma = self.add_weight(
            name = 'gamma',
            shape = (self.heads, d),
            initializer = tf.keras.initializers.ones()
        )

    def call(self, x):
        out = tf.einsum('...e, se -> ...se', x, self.gamma) + self.beta
        return tf.unstack(out, axis = -2) if self.split_heads else out
        

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "heads": self.heads,  
                "split_heads": self.split_heads,  
            }
        )
        return config
    

#----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
class ReLUSquared(tf.keras.layers.Layer):
    """
    ReLU Squared (ReLUSquared)
    https://arxiv.org/pdf/2104.07012.pdf

    They introduce a novel, simple method for achieving sparsity in attention, by
    replacing the softmax activation with ReLU, and show that sparsity naturally
    emerges from such a formulation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return tf.math.square(tf.nn.relu(x))

#----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
class LaplacianAttnFn(tf.keras.layers.Layer):
    """
    Laplacian Attention Function (LaplacianAttnFn)
    https://arxiv.org/abs/2209.10655

    Replacement for Squared ReLU via architecture search techniques which has
    shown faster convergence speed and competitive generalization performance
    on language tasks.
    """

    def __init__(self,
                 *,
                 use_n : bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_n = use_n
        self.PI = 3.14159265358

    def call(self, x):

        ## With the increasing length of the vectors the results are starting
        ## to become 1.0, to prevent this I introduced a new variable into
        ## the function to squeeze the domain a little bit. With this
        ## modification the values won't become 1.0 at the near end of the vecs.
        n = tf.saturate_cast(x.shape[-2], x.dtype) if self.use_n else 2.0

        mu = tf.saturate_cast(tf.math.sqrt(0.5), x.dtype)
        std = tf.saturate_cast(tf.math.sqrt(0.25 * self.PI), x.dtype)
        inner = (x - mu) / (std * tf.cast(tf.math.sqrt(n), x.dtype))
        return 0.5 * (1 + tf.math.erf(inner))

    def get_config(self):
        config = super().get_config()
        config.update({'use_n': self.use_n})
        return config
#----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------    
class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.n, self.num_token, self.dim = input_shape.as_list()

        self.scale = self.add_weight(
                name="scale",
                shape=(self.dim,),
                dtype=tf.dtypes.float32,
                initializer=tf.constant_initializer(value=1.0),
                trainable=True
        )
    def call(self, x):
        '''
        x : (b,num_token,dim)
        '''
        return x*self.scale

    def get_config(self):
        config = super(ScaleLayer, self).get_config()
        return config 
    

#----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------      

class ScaleNormLayer(tf.keras.layers.Layer):
    def __init__(self,
            ln_epsilon=1e-5,           
            **kwargs):
        super().__init__( **kwargs)
        self.eps = ln_epsilon

    def build(self, input_shape):

        self.n, self.num_token, self.dim = input_shape.as_list()

        self.scale_factor = tf.math.rsqrt(
            tf.cast(self.dim, dtype=tf.dtypes.float32)
        ).numpy()

        self.norm_scale = self.add_weight(
                    name="ScaleNorm_factor",
                    shape=None,
                    dtype=tf.dtypes.float32,
                    initializer=tf.constant_initializer(value=self.scale_factor),
                    trainable=False
        )
            
        self.g = self.add_weight(
                name="ScaleNorm_g",
                shape=None,
                dtype=tf.dtypes.float32,
                initializer=tf.constant_initializer(value=1.0),
                trainable=True
        )
        
    def call(self, x):
        '''
        x : (b,17,dim)
        '''
        norm = tf.norm(x, axis=-1, keepdims=True)*self.norm_scale  #(b,17,1)
        norm = tf.math.maximum(norm,self.eps) #(b,17,1)
        return (x/norm)*self.g 


    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "eps": self.eps,  
                "scale_factor": self.scale_factor,  
                "input_shape": (None, self.num_token, self.dim),  
            }
        )
        return config 
    
    
#----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------  

class RMSNormLayer(tf.keras.layers.Layer):
    r"""
    Root Mean Square Layer Normalization (RMSNorm)
    https://arxiv.org/pdf/1910.07467.pdf

    A well-known explanation of the success of LayerNorm is its re-centering
    and re-scaling invariance property. However RMSNorm only focuses on
    re-scaling invariance and regularizes the summed inputs simply according
    to the root mean square statistic.

    Intuitively, RMSNorm simplifies LayerNorm by totally removing the
    mean statistic at the cost of sacrificing the invariance that mean
    normalization affords.
    """

    def __init__(self,
                ln_epsilon: float = 1e-5,
                use_bias : bool = False,
                **kwargs):
        super().__init__(**kwargs)
        self.eps = ln_epsilon
        self.use_bias = use_bias

    def build(self, x_shape):
        d = x_shape[-1]

        self.scale = self.add_weight(
            name = 'scale',
            shape = (1, d),
            initializer = tf.ones_initializer())

        self.offset = self.add_weight(
            name = 'offset',
            shape = (1, d) if self.use_bias else (1,),
            initializer = tf.zeros_initializer())

        self.built = True

    def call(self, x):
        ms = tf.reduce_mean(tf.math.square(x), axis = -1, keepdims = True)
        return self.scale * x * tf.math.rsqrt(ms + self.eps) + self.offset

    def get_config(self):
        config = super().get_config()
        config.update({
            'eps': self.eps,
            'use_bias': self.use_bias
        })
        return config
#----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------  
@LAYERS.register_module()
class GatedAttentionUnit(tf.keras.layers.Layer):
    VERSION = '1.0.0'
 
    R""" GatedAttentionUnit

    Formulates the attention and Gated Linear Unit (GLU) as a unified layer and
    to share their computation as much as possible.

    First apply a ScaleNorm on the input 'x' and formulates the two gates
    'u' and 'v'. On the other hand computes the attention 'A = attn(x, v)',
    and apply an element-wise multiplication (u * A). Finally the dense layer
    which closes the inverse bottleneck and transforms the embeddings back to
    original size along feature axis.

    References:
        - [Based on implementation of 'RTMCCBlock' @mmpose] 
           (https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/rtmcc_block.py#L58)    
        - [Inspired by 'GatedAttentionUnit' in layers.py @bert4keras] 
           (https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L672)       
        - [Inspired by 'GAU' in gau_tensorflow.py @brandnewchoppa] 
           (https://github.com/brandnewchoppa/gau-tensorflow/blob/main/gau_tensorflow/gau_tensorflow.py#L165)
        - [Inspired by 'GAU' in FLASH-pytorch.py @lucidrains] 
           (https://github.com/lucidrains/FLASH-pytorch/blob/main/flash_pytorch/flash_pytorch.py#L158)

    Args:
        scaling_factor (float):  Default to 1
        relative_attention_num_buckets (int):  Sequence axis in the input tensor. Default to 1
        relative_attention_max_distance (int):  Feature axis in the input tensor. Default to -1
        bidirectional (bool):  Default to 0


    Examples:

    ```python
        from lib.Registers import MODELS
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        x = Input(shape=(17, 256))
        out = GatedAttentionUnit(    
            query_key_dim =128,
            expansion_factor = 2.,
            use_shortcut = True,
            use_bias = False,
            use_laplace_attn = False,
            use_rel_pos_bias = True,
            use_rope = True,
            name ='GAU'
        )(x)
        model = Model(x, out)
        model.summary(200)  
    """
    

    def __init__(self,
                out_token_dims : int,
                query_key_dims : int = 128,
                expansion_factor : float = 2.,
                dropout_rate : float  = 0.,
                ln_epsilon : float= 1e-5,
                norm_type : str = 'scale_norm',
                use_rel_pos_bias = False,
                use_bias = True,
                use_rope : bool = False,
                use_laplace_attn : bool = False,
                **kwargs):
        super().__init__(**kwargs)
        self.out_token_dims = out_token_dims
        self.expansion_factor = expansion_factor
        self.att_dims = query_key_dims
        self.norm_type = norm_type
        self.ln_epsilon = ln_epsilon
        self.dropout_rate = dropout_rate
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_bias = use_bias
        self.use_rope = use_rope
        self.use_laplace_attn = use_laplace_attn
        self.sqrt_s =self.att_dims**0.5
        self.use_shortcut = False

    def build(self,input_shape):
        """ 
        input_shape : (b, num_head, seq_q , seq_k) or (b, seq_q , seq_k) 
        """
        _, self.num_token, self.in_token_dims = input_shape
        if self.out_token_dims ==-1 :
            self.out_token_dims = self.in_token_dims 

        self.exapnd_dims = int(self.in_token_dims*self.expansion_factor)

        if self.norm_type == 'scale_norm' :
            self.norm = ScaleNormLayer(ln_epsilon = self.ln_epsilon)
        elif self.norm_type == 'rms_norm' :
            self.norm = RMSNormLayer(ln_epsilon = self.ln_epsilon)
        else:
            self.norm = LayerNormalization(epsilon=self.ln_epsilon) 

        self.dens_uv_qk = Dense(
            units=(self.exapnd_dims*2+self.att_dims), use_bias=self.use_bias
        )        
  
        self.dens_out = Dense(
			units=self.in_token_dims, use_bias=self.use_bias
		)
        self.scale_offset = OffsetScale(
            heads = 2,       
            split_heads = False,
            name='OffsetScale'
        )

        self.attn_fn = LaplacianAttnFn() if self.use_laplace_attn else ReLUSquared()
        self.mul = tf.keras.layers.Multiply()

        if self.dropout_rate >0. :
            self.attn_dropout = Dropout(self.dropout_rate)

        if  self.out_token_dims==self.in_token_dims:
            self.use_shortcut = True
            self.scale_shortcut = ScaleLayer()  
        
        if self.use_rope :
            self.rope = RotaryPositionEmbedding(       
                max_wavelength=10000,
                scaling_factor=self.in_token_dims**0.5,
                sequence_axis=1,
                feature_axis=-1,
                name ='rope'
            ) #scaling_factor = =self.in_token_dims**0.5  or 0.1 ????

        if self.use_rel_pos_bias :
           self.add_rel_pos_bias = AddRelativePositionBiasT5(    
                scaling_factor = self.in_token_dims ** 0.5,
                relative_attention_num_buckets= 32,
                relative_attention_max_distance= 128,
                bidirectional= False,
                name ='RelativePositionBias'
            ) #scaling_factor = =self.in_token_dims**0.5  or 0.1 ????
           


    def call(self, inputs):
        x = self.norm(inputs) #(b, num_token, in_dims)
        uv_qk = self.dens_uv_qk(x)  #(b, num_token, in_dims*2*2+att_dims) ,exapnd_dims=in_dims*2
        uv_qk = tf.nn.silu(uv_qk)  #(b, num_token, exapnd_dims*2+att_dims)
        u, v, qk  = tf.split(
            uv_qk,
            [self.exapnd_dims,self.exapnd_dims,self.att_dims] , 
            axis=-1
        )
        qk = self.scale_offset(qk) # q, k : (b,num_token, att_dims) or qk : (b,num_token, 2, att_dims)

        if hasattr(self,'rope'):
           qk = self.rope(qk)

        q, k =  tf.unstack(qk,axis=2)
        qk = tf.matmul(q, k, transpose_b=True)  # (b,num_token,att_dims)x(b,,att_dims,num_token)=>(b,num_token,num_token)
        
        if hasattr(self,'add_rel_pos_bias'):
          qk = self.add_rel_pos_bias(qk)

        attn =  self.attn_fn(qk/self.sqrt_s)  # (b,num_token,num_token)
      
        if hasattr(self,'attn_dropout'):
            attn = self.attn_dropout(attn)
            
        out = self.mul([ tf.matmul(attn,v), u])
        out = self.dens_out(out)  #(b,num_token,exapnd_dims) => (b,num_token,out_dims)

        if hasattr(self,'scale_shortcut'):
            out = out + self.scale_shortcut(inputs) 
            
        return  out
   
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "query_key_dim" : self.att_dims,
                "expansion_factor": self.expansion_factor,
                "norm_type" : self.norm_type,
                "ln_epsilon" : self.ln_epsilon,
                "dropout_rate" : self.dropout_rate,
                "use_bias": self.use_bias,
                "use_shortcut": self.use_shortcut,
                "use_rope" : self.use_rope,
                "use_rel_pos_bias": self.use_laplace_attn,
                "use_laplace_attn" : self.use_laplace_attn,
                "sqrt_s " : self.sqrt_s  
            }
        )
        return config
    