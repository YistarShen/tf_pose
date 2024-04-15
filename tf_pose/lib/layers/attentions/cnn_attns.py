'tf layers'
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization,LayerNormalization, Activation, DepthwiseConv2D, Dense, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import ZeroPadding1D, ZeroPadding2D, Concatenate, Multiply, Reshape
from tensorflow import Tensor
import tensorflow as tf
from tensorflow_addons.layers import AdaptiveAveragePooling2D

#-------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------
class SqueezeAndExcitation(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r"""SE SqueezeAndExcitation(se_module)
    se_module : https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/common_layers.py

    """

    def __init__(self,  
                se_ratio : float = 0.25,
                divisor : int= 8,
                limit_round_down : int= 0.9,
                use_bias : bool = False,
                use_conv : bool = True,
                hidden_activation = 'relu',
                output_activation = 'relu',  
                se_activation = 'sigmoid',    
                name : str='se_layer', **kwargs) ->tf.keras.layers.Layer:
        
        super(SqueezeAndExcitation, self).__init__(name=name)

        if not isinstance(hidden_activation,(str, type(None))):
            raise TypeError("activation must be 'str' type like 'relu'"
                         f"but got {type(hidden_activation)} @{self.__class__.__name__}"
            )
        
        if not isinstance(se_activation,(str, type(None))):
            raise TypeError("activation must be 'str' type like 'sigmoid'"
                         f"but got {type(se_activation)} @{self.__class__.__name__}"
            ) 

        if not isinstance(output_activation,(str, type(None))):
            raise TypeError("activation must be 'str' type like 'relu'"
                         f"but got {type(output_activation)} @{self.__class__.__name__}"
            ) 
        

        self.se_ratio = se_ratio
        self.divisor = divisor
        self.limit_round_down = limit_round_down
        self.use_bias = use_bias
        self.use_conv = use_conv
        self.se_act = 'sigmoid' if se_activation is None else se_activation
        self.hidden_act = hidden_activation
        self.output_act = output_activation

   
    def build(self, input_shape):

        _, h,w,self.filters = input_shape
        def make_divisible(vv, divisor=4, min_value=None, limit_round_down=0.9):
            """Copied from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py"""
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(vv + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < limit_round_down * vv:
                new_v += divisor
            return new_v
        
        self.gap = GlobalAveragePooling2D(keepdims=True, name='GAP')

        self.reduction = make_divisible(self.filters *self.se_ratio,  self.divisor, limit_round_down=self.limit_round_down)
        if self.use_conv:
            self.squeeze = Conv2D(self.reduction, kernel_size=1, use_bias=self.use_bias, activation=self.hidden_act, name="s_conv")
        else:
            self.squeeze = Dense(self.reduction, use_bias=self.use_bias, activation=self.hidden_act, name="s_dense")   

        # if self.hidden_act is not None :
        #     self.hidden_act =  Activation(self.hidden_act, name=self.hidden_act)

        if self.use_conv:
            self.excitation = Conv2D(self.filters, kernel_size=1, use_bias=self.use_bias,  activation = self.se_act, name="e_conv")
        else:
            self.excitation = Dense(self.filters, use_bias=self.use_bias,  activation = self.se_act, name="e_dense")

        #self.se_act =  Activation(self.se_act, name=self.se_act)   
        self.mul = Multiply(name="mul_out")

        if self.output_act is not None :
            self.output_act =  Activation(self.output_act, name=self.output_act+'_out') 

    def call(self,x):
  
        se = self.gap(x)
        se = self.squeeze(se)

        # if self.hidden_act is not None :
        #     se = self.hidden_act(se)

        se = self.excitation(se)
        #se = self.se_act(se)
        se = self.mul([x,se]) 

        # if self.output_act is not None :
        #     se = self.output_act(se)     
        return se if self.output_act is None else self.output_act(se)  
    
    def get_config(self):
        config = super(SqueezeAndExcitation,self).get_config()
        config.update(
                {
                'filters' : self.filters,
                'hidden_channels' : self.reduction,   
                "se_ratio": self.se_ratio,
                "limit_round_down": self.limit_round_down,
                "use_bias": self.use_bias,
                "use_conv": self.use_conv,
                "se_activation": self.se_act,
                "hidden_activation": self.hidden_act,
                "output_activation": self.output_act,

                }
        )
        return config
    

#-------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------
class GlobalContextAttention(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r"""Global Context Attention Block(global_context_module)
    global_context_module : https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/common_layers.py
    Global Context Attention Block, arxiv:  https://arxiv.org/pdf/1904.11492.pdf
    """
    def __init__(self,  
                use_attn : bool=True,
                ratio : float = 0.25,
                divisor : int= 1,
                use_bias : bool = True,
                hidden_activation : str='relu',
                output_activation : str='relu', 
                ln_epsilon : float= 1e-5,   
                name : str='se_layer', **kwargs) ->tf.keras.layers.Layer:
        
        super(GlobalContextAttention, self).__init__(name=name)

        if not isinstance(hidden_activation,str):
            raise TypeError("activation must be 'str' type like 'relu'"
                         f"but got {type(hidden_activation)} @{self.__class__.__name__}"
            )

        if not isinstance(output_activation,str):
            raise TypeError("activation must be 'str' type like 'relu'"
                         f"but got {type(output_activation)} @{self.__class__.__name__}"
            ) 
        

        self.ratio = ratio
        self.divisor = divisor
        self.use_attn = use_attn
        self.use_bias = use_bias
        self.hidden_act = hidden_activation
        self.output_act = output_activation
        self.ln_epsilon = ln_epsilon

   
    def build(self, input_shape):

        _, self.h,self.w,self.filters = input_shape
        def make_divisible(vv, divisor=4, min_value=None, limit_round_down=0.9):
            """Copied from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py"""
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(vv + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < limit_round_down * vv:
                new_v += divisor
            return new_v
        
 

        self.reduction = make_divisible(self.filters *self.ratio,  self.divisor, limit_round_down=0.0)
        if self.use_attn:
            self.attn_conv = Conv2D(1, kernel_size=1, use_bias=self.use_bias, name="attn_conv")
            #self.attn_reshape = Reshape(attn, [-1, 1, height * width]) 
            self.attn_act = Activation('softmax', name='softmax')
        else:
            self.gap = GlobalAveragePooling2D(keepdims=True, name='GAP')

        self.mlp_1_conv = Conv2D(self.reduction, kernel_size=1, use_bias=self.use_bias, name="mlp_1_conv")
        self.mlp_ln = LayerNormalization(epsilon=self.ln_epsilon, name='ln')
        self.mlp_hidden_act = Activation(self.hidden_act, name=self.hidden_act)
        self.mlp_2_conv = Conv2D(self.filters , kernel_size=1, use_bias=self.use_bias, name="mlp_2_conv")
        self.mlp_output_act = Activation(self.output_act, name=self.output_act)



    def call(self,x):

        if self.use_attn:
            attn = self.attn_conv(x) #(b,h,w,c)->(b,h,w,1)
            attn = Reshape((1,self.h*self.w), name='attn_flatten')(attn) #(b, 1, h*w)
            attn = self.attn_act(attn)
            context  = Reshape((self.h*self.w, -1), name='attn_flatten')(x) #(b,h,w,c)->(b,h*w,c)
            context = tf.matmul(attn, context)  # (b,1, h*w)x(b,h*w,c) =>  (b,1,c)
            context  = Reshape((1,1,-1), name='attn_expand')(context) #(b,1,c)->(b,1,1, c)
        else:
            context = self.gap(x)

        mlp = self.mlp_1_conv(x)   #(b,1,1,c)->(b,1,1,hidden_c)
        mlp = self.mlp_ln(mlp)
        mlp = self.mlp_hidden_act(mlp)
        mlp = self.mlp_2_conv(mlp)  #(b,1,1,c)->(b,1,1,c)
        mlp = self.mlp_output_act(mlp)
        return Multiply(name="mul_out")([x, mlp])

    def get_config(self):
        config = super(GlobalContextAttention,self).get_config()
        config.update(
                {
                'filters' : self.filters,
                'hidden_channels' : self.reduction,   
                "ratio": self.ratio,
                "divisor": self.divisor,
                "use_bias": self.use_bias,
                "use_attn": self.use_attn,
                "hidden_activation": self.hidden_act,
                "output_activation": self.output_act,
                "ln_epsilon": self.ln_epsilon
                }
        )
        return config
    
#-------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------
class EfficientChannelAttention(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r"""Efficient Channel Attention Block(eca_module)
    eca_module : https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/common_layers.py
    Efficient Channel Attention Block, arxiv: https://arxiv.org/pdf/1910.03151.pdf
    """
    def __init__(self,  
                gamma : float= 2.0, 
                beta : float=1.0,
                **kwargs) ->tf.keras.layers.Layer:
        
        super().__init__(**kwargs)
        self.gamma = gamma
        self.beta = beta
        
    def build(self, input_shape):
        _, _,_,self.filters = input_shape
        tt = int(( tf.math.log(float(self.filters)) / tf.math.log(2.0) + self.beta) / self.gamma)
        self.kernel_size = max(tt if tt % 2 else tt + 1, 3)
        self.pad_size =  self.kernel_size  // 2

        self.gap = GlobalAveragePooling2D(
            keepdims=False, name='GAP'
        )
        self.conv1d = Conv1D(
            1, kernel_size=self.kernel_size, strides=1, padding="valid", use_bias=False, name= "conv1d"
        )
        self.attn_act = Activation(
            'sigmoid', name='sigmoid'
        )

    def call(self, inputs):

        nn = self.gap(inputs) #(b,c)
        if self.pad_size :
            nn = tf.pad(
                nn, [[0, 0], [ self.pad_size,  self.pad_size]]
            )  #(b,c')
       
        #nn = ZeroPadding1D(self.pad_size, name='pad')(nn)  #(b,c')
        nn = tf.expand_dims(nn, axis=-1, name='ops_expand_dims')  #(b,c',1)   
        nn = self.conv1d(nn) #(b,c,1)
        nn = tf.squeeze(nn, axis=-1, name='ops_squeeze') #(b,c,1)->(b,c)
        nn = self.attn_act(nn)  #(b,c)
        nn = tf.reshape(nn,[-1,1,1,self.filters])
        #nn = nn[:,None, None, :]
        return  Multiply(name="mul_out")([inputs, nn])
    
    def get_config(self):
        config = super().get_config()
        config.update(
                {
                'filters' : self.filters,
                'kernel_size_1D' : self.kernel_size,   
                "pad_size": self.pad,
                }
        )
        return config   