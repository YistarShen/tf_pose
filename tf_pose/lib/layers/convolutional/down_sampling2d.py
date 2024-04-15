from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tensorflow import Tensor
import tensorflow as tf
from tensorflow.keras.layers import  ZeroPadding2D, MaxPooling2D, AveragePooling2D, Concatenate
from lib.layers import Conv2D_BN


#----------------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------------
class MaxPoolAndStrideConv2D(tf.keras.layers.Layer):

    VERSION = '1.0.0'
    r"""MaxPoolAndStrideConv2D ( MaxPooling and Conv2D with strides=2 )

    advanced downsample bolok that fuse two diiferent features (one is from MaxPooling2D and another is from conv)
    this downsample block can get more diversity to improve performance compared with conv with stride=2 and kernel=3

    it was used in YoloV7'backbone to down sample

    Architecture :
        inputs = Input(shape=(80,80,256))
        MaxPool_Conv2D = {
            strides = 2,
            use_bias = False,
        }

        [-1, 1, Conv, [256, 1, 1]]              #arbitrary dummy input with shape (40,40,256)
        [-1, 1, MP, []],                        #B1_mp    (1/2 downsample due to strides=2)
        [-1, 1, Conv, [128, 1, 1]],             #B1_conv_pw
        [-3, 1, Conv, [128, 1, 1]],             #B2_conv_pw
        [-1, 1, Conv, [128, 3, 2]],             #B2_conv  (1/2 downsample due to strides=2)
        [[-1, -3], 1, Concat, [-1]],            #Output_channels = dummy_input_channels and shape is (20,20,256)


    References:
            - [Based on implementation of 'concat_stack'BrokenPipeError @leondgarse's keras_cv_attention_models  ] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/yolov7/yolov7.py)
    Args:
            in_channels_ratio(float) : ratio of input tensor channels to in palnes(hidden cahnnels), defaults to 0.5.
            strides (int) : downsample factor, 2 means downdample to 1/2, defaults to 2
            use_bias (bool) :  whether use bias of conv, defaults to False
            bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
            bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
            activation (str) :  activation used in conv_bn blocks, defaults to 'relu'.
            name (str) :'MP_Conv2D'
    
    Note: 
        YoloV7Tiny -  { stack_depth=4 ,  mid_ratio=1.0, stack_concats= [-1, -2,-,3,-4] } in backbone


    """
    def __init__(self,
                in_channels_ratio : float = 0.5, 
                strides : int =2,
                use_bias : bool =False,
                bn_epsilon : float= 1e-5,
                bn_momentum : float= 0.9,
                activation = 'relu',
                name='MP_Conv2D',  **kwargs):
        
        super(MaxPoolAndStrideConv2D, self).__init__(name=name)
        self.ratio = in_channels_ratio
        self.strides = strides
        self.use_bias = use_bias
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.act_name = activation
        
    def build(self, input_shape):
        b, h, w, c = input_shape

        self.in_planes =  int(c*self.ratio)
        
        self.B1_mp = MaxPooling2D((2, 2), strides=(self.strides, self.strides), name='B1_mp')

        self.B1_conv_pw = Conv2D_BN(filters =  self.in_planes,
                                    kernel_size=1,
                                    strides=1,
                                    use_bias = self.use_bias,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum = self.bn_momentum,
                                    activation = self.act_name,
                                    name='B1_conv_pw')
        
        self.B2_conv_pw = Conv2D_BN(filters =  self.in_planes,
                                    kernel_size=1,
                                    strides=1,
                                    use_bias = self.use_bias,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum = self.bn_momentum,
                                    activation = self.act_name,
                                    name='B2_conv_pw')
        

        self.B2_conv = Conv2D_BN(filters =  self.in_planes,
                                kernel_size=3,
                                strides=self.strides,
                                use_bias = self.use_bias,
                                bn_epsilon = self.bn_epsilon,
                                bn_momentum = self.bn_momentum,
                                activation = self.act_name,
                                name='B2_conv')   

        self.concat = Concatenate(axis=-1, name='concat')          

    def call(self, x : Tensor):
        #self.build((1,256,256,128))
        x1 = self.B1_mp(x) 
        x1 = self.B1_conv_pw(x1) 
        x2 = self.B2_conv_pw(x) 
        x2 = self.B2_conv(x2) 
        out = self.concat([x1, x2])
        return out

    def get_config(self):
        config = super(MaxPoolAndStrideConv2D, self).get_config()
        config.update(
                {
                "in_planes": self.in_planes,
                "strides": self.strides,
                "use_bias": self.use_bias,
                "bn_epsilon": self.bn_epsilon,
                "bn_momentum": self.bn_momentum,
                "act": self.act_name
                }
        )
        return config  
    
#----------------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------------
class ADown(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r""" SP MaxPool2d with input Padding


    """
    def __init__(
            self, 
            out_channels : int ,
            strides : int =2,
            bn_epsilon : float= 1e-5,
            bn_momentum : float= 0.9,
            activation = 'relu',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.strides = strides
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.act_name = activation

    def build(self, input_shape):

        if self.out_channels < 0 :
            _, _, _, in_channels = input_shape
            self.out_channels = in_channels
  

        self.conv1 = Conv2D_BN(
            filters = self.out_channels//2,
            kernel_size=3,
            strides=self.strides,
            activation = self.act_name,
            bn_epsilon = self.bn_epsilon,
            bn_momentum = self.bn_momentum,
            name = 'conv1'
        ) 

        self.conv2 = Conv2D_BN(
            filters = self.out_channels//2,
            kernel_size=1,
            strides=1,
            activation = self.act_name,
            bn_epsilon = self.bn_epsilon,
            bn_momentum = self.bn_momentum,
            name = 'conv2'
        )   

        self.split_layer = tf.keras.layers.Lambda(
            lambda x : tf.split(x,  num_or_size_splits=2, axis=-1), 
            name = 'split'
        )

        self.MaxPooling2D = MaxPooling2D(
                (3,3) , 
                strides= self.strides, 
                padding="same", 
                name=f'max_pool2d'
        )
        self.AvgPooling2D = AveragePooling2D(
                (2,2) , 
                strides=1, 
                padding="same", 
                name=f'avg_pool2d'
        )
        self.concat = Concatenate(axis=-1, name='concat')

    def call(self, x : Tensor):
        x = self.AvgPooling2D(x)
        x1, x2 = self.split_layer(x) 
        x1 = self.conv1(x1)
        x2 = self.MaxPooling2D(x2)
        x2 = self.conv2(x2)
        return self.concat([x1,x2]) 
    
#----------------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------------
class SP(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r""" SP MaxPool2d with input Padding

    """
    def __init__(
            self, 
            pool_size : Tuple[int] = (2, 2),
            strides : int = 1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides

    def build(self, input_shape):

        _, _, _, self.in_channels = input_shape
        padding = (self.pool_size[0]//2, self.pool_size[1]//2)
        if max(padding)>0:
            self.zero_padding2D = ZeroPadding2D(padding=padding, name='pad')

        self.MaxPooling2D = MaxPooling2D(
                self.pool_size , 
                strides=self.strides, 
                padding="valid", 
                name=f'pool2d'
        )
    def call(self, x : Tensor):
        if hasattr(self,'zero_padding2D'):
            x = self.zero_padding2D(x)
    
        return self.MaxPooling2D(x)   
    