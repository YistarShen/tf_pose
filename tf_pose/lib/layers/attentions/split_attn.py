import tensorflow as tf
from tensorflow.keras.layers import Activation, Multiply, Conv2D, GlobalAveragePooling2D, Reshape, Multiply, Permute
from lib.layers import Conv2D_BN
from typing import Optional
class SplitAttention(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    R""" SplitAttention used in ResNest


    #https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/volo/volo.py
    #https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/models/splat.py


    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    #print(out.shape)
    x = Input(shape=(32,32,128))
    out = SplitAttention(
        out_channels= 128,
        kernel_size  = 3, 
        strides = 1,
        radix  = 4,
        groups = 1,
        bn_epsilon = 1e-5,
        bn_momentum = 0.9,   
        name='SplitAttention'
    )(x)

    model = Model(x, out)
    model.summary(150)

    """
    def __init__(
            self, 
            out_channels : Optional[int] = None,
            kernel_size : int = 1, 
            strides : int = 1,
            radix : int = 4,
            groups : int = 1,
            bn_epsilon : float= 1e-5,
            bn_momentum : float= 0.9,
            **kwargs 
    ):
        super().__init__(**kwargs)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.radix = radix
        self.groups = groups
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum


    def build(self, input_shape):

        _, self.H, self.W, self.c  = input_shape
        if self.out_channels is  None or self.out_channels<0:
            self.out_channels = self.c 

        
        self.g_conv = Conv2D_BN(
            filters = self.out_channels*self.radix,
            kernel_size=self.kernel_size ,
            strides = self.strides,
            groups = self.groups**self.radix,
            bn_epsilon = self.bn_epsilon,
            bn_momentum = self.bn_momentum,
            activation = 'relu',
            name = 'g_ConvBn'
        )


        reduction_factor = 4
        hidden_channels = max(self.c*self.radix //reduction_factor, 32)
        
        self.fc1_bn = Conv2D_BN(
            filters = hidden_channels,
            kernel_size=1 ,
            strides = 1,
            use_bias= True , 
            bn_epsilon = self.bn_epsilon,
            bn_momentum = self.bn_momentum,
            groups = self.groups,
            activation = 'relu',
            name = 'fc1'
        )  

        self.fc2 = Conv2D(
            self.out_channels*self.radix,
            kernel_size=1,
            strides =1,
            use_bias= True , 
            groups = self.groups,
            name="fc2"
        )

        

    # def _rsoftmax(self, inputs, radix):
    #     if radix > 1:
    #         #(b,1,1, c*radix)
    #         nn = tf.reshape(inputs, [-1, radix, inputs])
    #         # nn = tf.transpose(nn, [0, 2, 1, 3])
    #         nn = tf.nn.softmax(nn, axis=1)
    #         nn = tf.reshape(nn, [-1, *inputs.shape[1:]])
    #     else:
    #         nn = Activation("sigmoid")(inputs)
    #     return nn


    def call(self, x) :

        x = self.g_conv(x) #(b,h,w,c*radix)


        gap = Reshape(
                [*(x.shape[1:3]), -1, self.radix], name="reshape_gap"
        )(x) #(b,h,w,c*radix) => (b,h,w,c, radix)
        gap = tf.reduce_sum(gap, axis=-1)  #(b,h,w,c)


        # splited = tf.split(
        #     x, 
        #     num_or_size_splits=self.radix, 
        #     axis=-1
        # )  # list type : [(b,h,w, c)]*radix

        # gap = Add(name='add')(splited) # (b,h,w,c)

        gap = GlobalAveragePooling2D(name='GAP', keepdims=True)(gap) #(b,1,1,c)
        gap = self.fc1_bn(gap)  #(b,1,1, hidden_channels), hidden_channels=c*radix/reduction_factor
        atten = self.fc2(gap) #(b,1,1, c*radix)

        """ 
        rsoftmax
        """
        if self.radix > 1:
            atten = Reshape(
                [self.groups, self.radix, -1], name="reshape1_rsoftmax"
            )(atten)  #(b,1,1, c*radix) => (b, groups, radix, c/groups) 
            atten = Permute([2,1,3], name="permute_rsoftmax")(atten) #(b, radix, groups, c/groups) 
            atten = tf.nn.softmax(atten, axis=1) #(b, radix, groups, c/groups) 
            atten = Reshape(
                [1,1,-1], name="reshape2_rsoftmax"
            )(atten) #(b,1,1, radix*c) 
        else:
            atten = Activation("sigmoid")(atten) 

        # x :  (b,h,w,c*radix)
        # atten :  (b,1,1, c*radix)
        output = Multiply(
            name='mul'
        )([atten, x])# atten :  (b,h,w, c*radix)

        output = Reshape(
                [self.H, self.W, self.radix, self.c], name="reshape1_attn"
        )(output) 
        
        output = tf.reduce_sum(output, axis=-2)
        return  output


            
    def get_config(self):
        config = super().get_config()
        config.update(
                {
                "out_channels"  : self.out_channels,
                'kernel_size' : self.kernel_size, 
                "strides": self.strides,
                }
        )
        return config 