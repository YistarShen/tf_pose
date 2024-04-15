
import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D,  DepthwiseConv2D, Conv2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, concatenate, Input, Activation
from tensorflow.keras.models import Model

from .base_cnn import Conv2D_BN_SiLU, SeparableConv2D_BN_SiLU
from .se_layer import ChannelAttention

############################################################################
#
#
############################################################################
class CSPNeXtBlock(tf.keras.layers.Layer):
    version = '1.0.0'
    r"""CSPNeXtBlock

    """
    def __init__(self,
                filters, 
                expansion = 0.5,
                add_identity = True,
                use_depthwise  = False,
                kernel_size = 5,              
                use_bias=False, 
                name=None, 
                **kwargs):
        super(CSPNeXtBlock, self).__init__(name=name)
        "SeparableConv2D_BN_SiLU"
        self.filters = filters
        self.expansion = expansion

        self.add_identity = add_identity


        self.use_depthwise  = use_depthwise
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.bn_m = 0.97
        self.bn_eps = 0.001

    def build(self, input_shape):
        '''
        if self.use_depthwise :
        self.conv = SeparableConv2D_BN_SiLU(filters=40, 
                                            kernel_size=(1,1), 
                                            strides=(1, 1), 
                                            use_bias=False)
        filters, kernel_size=(1,1), strides=(1, 1), use_bias=False, 
        else: 
        self.conv = Conv2D_BN_SiLU(filters=self.filters, 
                                    kernel_size=(3,3), 
                                    strides=(1, 1), 
                                    use_bias=self.use_bias)
        '''
        conv = SeparableConv2D_BN_SiLU if self.use_depthwise else Conv2D_BN_SiLU

        self.conv1 = conv(filters=self.filters*self.expansion, 
                        kernel_size=(3,3), 
                        strides=(1, 1), 
                        use_bias=self.use_bias)
        
        self.conv2 = SeparableConv2D_BN_SiLU(filters=self.filters, 
                                            kernel_size=self.kernel_size, 
                                            strides=(1, 1), 
                                            use_bias=self.use_bias)

    def call(self, x):

        if ( self.add_identity and x.shape[-1]==self.filters ) :
            self.add_identity = True
            identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        out = x + identity if self.add_identity else x
        return out

    def get_config(self):
        config = super(CSPNeXtBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "expansion" : self.expansion,
                "add_identity" : self.add_identity,
                "use_depthwise" : self.use_depthwise,
                "kernel_size": self.kernel_size,
                "use_bias": self.use_bias,            
            }
        )
        return config 
    
############################################################################
#
#
############################################################################
class CSPLayer(tf.keras.layers.Layer):
    version = '1.0.0'
    r"""CSPLayer

    """
    def __init__(self,
                filters, 
                expand_ratio = 0.5,
                num_blocks = 1,
                add_identity = True,
                use_depthwise  = False,
                channel_attention = False,
                name=None, 
                **kwargs):
        super(CSPLayer, self).__init__(name=name)
        "SeparableConv2D_BN_SiLU"
        self.filters = filters
        self.expansion = expand_ratio
        self.num_blocks = num_blocks
        self.add_identity = add_identity
        self.use_depthwise  = use_depthwise
        self.channel_attention  = channel_attention

        self.mid_channels = int(self.filters * self.expansion)

    def build(self, input_shape):
        self.n, self.h, self.w, self.c = input_shape.as_list()

        self.main_conv = Conv2D_BN_SiLU(filters = self.mid_channels, 
                        kernel_size = (1,1), 
                        strides = (1, 1), 
                        use_bias  = False)
        
        self.short_conv = Conv2D_BN_SiLU(filters = self.mid_channels, 
                        kernel_size = (1,1), 
                        strides = (1, 1), 
                        use_bias  = False)
        
        self.final_conv = Conv2D_BN_SiLU(filters = self.filters , 
                        kernel_size = (1,1), 
                        strides = (1, 1), 
                        use_bias  = False)
        
        self.blocks = tf.keras.Sequential([ CSPNeXtBlock(filters = self.mid_channels, 
                                    expansion = 1., 
                                    kernel_size = 5, 
                                    add_identity = self.add_identity, 
                                    use_depthwise = self.use_depthwise) for _ in range( self.num_blocks) ] )

        if self.channel_attention :
            self.attention = ChannelAttention(self.filters)
    

    def call(self, x):
        
        x_short = self.short_conv(x)
        x = self.main_conv(x)
        x = self.blocks(x)
        out = tf.keras.layers.concatenate([x,x_short], axis=-1)

        if self.channel_attention:
            out = self.attention(out)

        out = self.final_conv(out)
        return out

    def get_config(self):
        config = super(CSPLayer, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "expansion" : self.expansion,
                "num_blocks" :    self.num_blocks,
                "add_identity" : self.add_identity,
                "use_depthwise" : self.use_depthwise,
                "use_channel_attention": self.channel_attention,        
            }
        )
        return config 