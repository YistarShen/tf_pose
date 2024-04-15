'tf layers'
from tensorflow.keras.layers import Conv2D, LayerNormalization, Activation
import tensorflow as tf
from tensorflow_addons.layers import AdaptiveAveragePooling2D
class PolarizedSelfAttention(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r""" tf_PSA(Polarized Self-Attention)
    it can be used in basic res_block in hrnet to improve accuracy

    Return : 
        tf.Tensor

    References:
        - [Paper : Polarized Self-Attention] 
           (https://paperswithcode.com/paper/polarized-self-attention-towards-high-quality-1)
        - [Based on  implementation of 'PolarizedSelfAttention.py' @xmu-xiaoma666/] 
           (https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/PolarizedSelfAttention.py)
      
    Args:
        inplanes (int): out_channels, defaults to -1 that means inplanes=in_channels=out_channels
        ln_epsilon (float) : epsilon of layer normalization , defaults to 1e-5.
        mode (str) : "p"(Parallel Type) or "s"(series type)
        name (str) : 'PSA'


    Note:

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
                inplanes : int = -1, 
                mode : str="p",                
                ln_epsilon : float= 1e-5,
                **kwargs ):
        super().__init__(**kwargs)
        if (not isinstance(mode,str)) or not (mode.lower() == 'p' or mode.lower() == 's'): 
            raise ValueError(f"the mode in PSA must be 'p' or's' "
                            f"but got mode: {mode}  @{self.__class__.__name__}"
            )
        
        if (not isinstance(ln_epsilon,(float))):
            raise TypeError(f"ln_epsilon must be 'float' type @{self.__class__.__name__}"
                         f"but got eps:{type(ln_epsilon)}"
            )

        self.mode = mode.lower()
        self.inplanes = inplanes
        self.ln_epsilon = ln_epsilon
   

    def build(self, input_shapes):
        _, self.h, self.w, self.c = input_shapes.as_list()

        if self.inplanes <0 :
            self.inplanes = self.c
        
        channel = self.inplanes//2

        'channel-only self-Attention (COSA)'
        self.conv_q_ch = Conv2D(filters=1, kernel_size=(1,1), strides=(1,1),
                                padding='same', use_bias=False,
                                kernel_initializer='he_normal',
                                name=self.name+"ch-conv_q")
        
        self.conv_v_ch = tf.keras.layers.Conv2D(channel, kernel_size=(1,1), strides=(1,1),
                          padding='same', use_bias=False,
                          kernel_initializer='he_normal',
                          name = self.name+"ch-conv_v")    
        
        self.softmax_ch = Activation('softmax', name = self.name+"ch-softmax")

        self.conv_ch = Conv2D(self.inplanes, kernel_size=(1,1), strides=(1,1),
                                padding='same', use_bias=False,
                                kernel_initializer='he_normal',
                                name = self.name+f"ch-conv")
        
        self.ln_ch = LayerNormalization(axis=[1,2,3], epsilon=self.ln_epsilon,
                                        name=self.name+"ln_s")


        self.sigmoid_ch  = Activation('sigmoid', name = self.name+"ch-sigmoid")
        self.layersMultiply_ch = tf.keras.layers.Multiply(name= self.name+ 'ch-Mul_out')     

        'spatial-only self-Attention'
        self.conv_q_sp = Conv2D(channel, kernel_size=(1,1), strides=(1,1),
                            padding='same', use_bias=False,
                            kernel_initializer='he_normal',
                            name = self.name+"sp-conv_q")
        self.conv_v_sp = Conv2D(channel, kernel_size=(1,1), strides=(1,1),
                            padding='same', use_bias=False,
                            kernel_initializer='he_normal',
                            name = self.name+"sp-conv_v")
        
        #self.gap = tf.keras.layers.GlobalAveragePooling2D(name = self.name+"sp-GAP",keepdims=True)
        self.agp = AdaptiveAveragePooling2D((1,1), name = self.name+"sp-AdaptAvgPool",)
        self.softmax_sp = Activation('softmax',  name = self.name+"sp-softmax")
        self.sigmoid_sp  = Activation('sigmoid', name = self.name+"sp-sigmoid")
        self.layersMultiply_sp = tf.keras.layers.Multiply(name= self.name+ 'sp-Mul_out')   

    def call(self, x):

        'channel-only self-Attention (ch) --spatial_pool'
        input_q = self.conv_q_ch(x) #(b,h,w,1)
        input_q = tf.keras.layers.Reshape((1, 1, self.h*self.w),name=self.name+"ch-reshape_1" )(input_q)  #(b,1,1,h*w)
        input_q = self.softmax_ch(input_q) #(b,1,1,h*w)

        input_v = self.conv_v_ch(x) #(b,h,w,c/2)
        input_v = tf.keras.layers.Reshape((1, self.h*self.w, self.c//2),name=self.name+"ch-reshape_2" )(input_v)  #(b,1,h*w,c/2)
        context = tf.matmul(input_q, input_v)  #(b,1,1,h*w)x(b,1,h*w,c/2)=> (b,1,1,c/2)
        context = self.conv_ch(context)
        context = self.ln_ch(context)
        mask_ch = self.sigmoid_ch(context) #(b,1,1,c)
        context_channel_out = self.layersMultiply_ch([x, mask_ch])  #(b,h,w,c)*(b,1,1,c)=>(b,h,w,c)

        'spatial-only self-Attention (sp)'
        if self.mode=='p':
            input_sp = x
        else :
            input_sp = context_channel_out
        input_q_sp = self.conv_q_sp(input_sp) #(b,h,w,c/2)

        input_q_sp = self.agp(input_q_sp) #(b,h,w,c/2) => (b,1,1,c/2)
        input_q_sp = self.softmax_sp(input_q_sp) #(b,1,1,c/2)

        input_v_sp = self.conv_v_sp(input_sp) #(b,h,w,c/2)
        input_v_sp = tf.keras.layers.Reshape((1, self.h*self.w, self.c//2),name=self.name+"sp-reshape_1" )(input_v_sp)  #(b,h,w,c/2)=>(b,1,h*w,c/2)
        input_v_sp = tf.keras.layers.Permute((1, 3, 2),name=self.name+"sp-permute_1")(input_v_sp)  #(b,1,h*w,c/2) =>(b,1,c/2,h*w)

        context = tf.matmul(input_q_sp, input_v_sp)  #(b,1,1,c/2)x(b,1,c/2,h*w)=> (b,1,1,h*w)
        context = tf.keras.layers.Reshape((self.h, self.w,1),name=self.name+"sp-reshape_2" )(context) #(b,1,1,h*w) => (b,h,w,1)
        mask_sp = self.sigmoid_sp(context) #(b,h,w,1)
        context_spatial_out = self.layersMultiply_sp([input_sp, mask_sp])  #(b,h,w,c)*(b,h,w,1)=>(b,h,w,c)

        if self.mode=='p':
            PSA_out = context_channel_out + context_spatial_out
        else :
            PSA_out = context_spatial_out

        return PSA_out
    
    def get_config(self):
        config = super().get_config()
        config.update(
                {
                "in_planes": self.inplanes,
                "mode": self.mode,
                "ln_epsilon": self.ln_epsilon,
                }
        )
        return config