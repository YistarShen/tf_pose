import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D,  DepthwiseConv2D, Conv2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, concatenate, Input, Activation
from tensorflow.keras.models import Model


#---------------------------------------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------------------------------------
class MP_Conv2DBNSiLU(tf.keras.layers.Layer):
    def __init__(self, transition_channels, use_bias=False, name=None, **kwargs):
        super(MP_Conv2DBNSiLU, self).__init__(name=name)
        self.filters = ( transition_channels//2 )

    def build(self, input_shape):
        self.B1_mp = MaxPooling2D((2, 2), strides=(2, 2))
        self.B1_conv_pw = Conv2D_BN_SiLU(self.filters, kernel_size=(1,1), strides=(1, 1), use_bias=False, name=None)
        self.B2_conv_pw = Conv2D_BN_SiLU(self.filters, kernel_size=(1,1), strides=(1, 1), use_bias=False, name=None)
        self.B2_conv = Conv2D_BN_SiLU(self.filters, kernel_size=(3,3), strides=(2, 2), use_bias=False, name=None)


    def call(self, x):
        x1 = self.B1_mp(x) 
        x1 = self.B1_conv_pw(x1) 
        x2 = self.B2_conv_pw(x) 
        x2 = self.B2_conv(x2) 
        out = concatenate([x1, x2],axis=-1)
        return out

    def get_config(self):
        config = super(MP_Conv2DBNSiLU, self).get_config()
        config.update(
                {
                "filters": self.filters,
                }
        )
        return config  

#---------------------------------------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------------------------------------
class Conv2D_BN_SiLU(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size=(1,1), strides=(1, 1), use_bias=False, name=None, **kwargs):
    super(Conv2D_BN_SiLU, self).__init__(name=name)
    "Conv2D_BN_SiLU"
    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.use_bias = use_bias
    self.bn_m = 0.97
    self.bn_eps = 0.001

  def build(self, input_shape):
    self.conv = Conv2D(self.filters, 
                self.kernel_size, 
                strides=self.strides, 
                padding='same', 
                use_bias=False)
    self.bn = BatchNormalization(epsilon=self.bn_eps, momentum=self.bn_m)
    
  def call(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = tf.nn.silu(x)
    return x

  def get_config(self):
    config = super(Conv2D_BN_SiLU, self).get_config()
    config.update(
            {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "use_bias": self.use_bias,
            }
    )
    return config  
  


#---------------------------------------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------------------------------------
class SeparableConv2D_BN_SiLU(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size=(1,1), strides=(1, 1), use_bias=False, name=None, **kwargs):
    super(SeparableConv2D_BN_SiLU, self).__init__(name=name)
    "SeparableConv2D_BN_SiLU"
    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.use_bias = use_bias
    self.bn_m = 0.97
    self.bn_eps = 0.001

  def build(self, input_shape):
    self.conv = tf.keras.layers.SeparableConv2D(self.filters, 
                                                self.kernel_size, 
                                                strides=self.strides, 
                                                padding='same', 
                                                use_bias=False)
    self.bn = BatchNormalization(epsilon=self.bn_eps, momentum=self.bn_m)
    
  def call(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = tf.nn.silu(x)
    return x

  def get_config(self):
    config = super(SeparableConv2D_BN_SiLU, self).get_config()
    config.update(
            {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "use_bias": self.use_bias,
            }
    )
    return config  
