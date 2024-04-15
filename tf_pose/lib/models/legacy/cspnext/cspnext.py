
import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D,  DepthwiseConv2D, Conv2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, concatenate, Input, Activation
from tensorflow.keras.models import Model

from .base_cnn import SeparableConv2D_BN_SiLU, Conv2D_BN_SiLU
from lib.models.backbones.base_backbone import BaseBackbone
from lib.Registers import MODELS
############################################################################
#
#
############################################################################
class CSPNeXtBlock(tf.keras.layers.Layer):
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
class ChannelAttention(tf.keras.layers.Layer):
  def __init__(self, channels, name=None, **kwargs):
    super(ChannelAttention, self).__init__(name=name)
    "ChannelAttention"
    self.channels = channels

  def build(self, input_shape):

    self.n, self.h, self.w, self.c = input_shape.as_list()

    self.gap = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
    self.fc =  tf.keras.layers.Conv2D(self.c , (1,1), strides=(1,1), padding='same', use_bias=False)
    self.act = tf.keras.layers.Activation('hard_sigmoid')
    self.mul = tf.keras.layers.Multiply()

  def call(self, x):
    assert x.shape[-1]==self.channels," channels must be x.shape[-1] @ChannelAttention"
    att = self.gap(x)
    att = self.fc(att)
    att = self.act(att)
    out =  self.mul([x,att])
    return out

  def get_config(self):
    config = super(ChannelAttention, self).get_config()
    config.update(
            {
            "channels": self.channels,
            "activation" : 'hard_sigmoid',
            }
    )
    return config  
############################################################################
#
#
############################################################################
class CSPLayer(tf.keras.layers.Layer):
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
  

############################################################################
#
#
############################################################################
class SPPBottleneck(tf.keras.layers.Layer):
  def __init__(self,
          filters, 
          pool_sizes=5,
          name=None, 
          **kwargs):
    super(SPPBottleneck, self).__init__(name=name)
    "SeparableConv2D_BN_SiLU"
    self.filters = filters
    self.pooling_sizes = pool_sizes

  def build(self, input_shape):
    self.n, self.h, self.w, self.c = input_shape.as_list()
    mid_channels = self.c//2
    self.input_conv = Conv2D_BN_SiLU(filters = mid_channels, 
                      kernel_size = (1,1), 
                      strides = (1,1), 
                      use_bias  = False)
    
    self.output_conv = Conv2D_BN_SiLU(filters = self.filters, 
                      kernel_size = (1,1), 
                      strides = (1,1), 
                      use_bias  = False)
    
    if isinstance(self.pooling_sizes, int):
      self.MaxPooling2D = MaxPooling2D(pool_size=(self.pooling_sizes, self.pooling_sizes), strides=(1, 1), padding='same')
    else :
      self.MaxPooling2D_list = [MaxPooling2D(pool_size=(m, m), strides=(1, 1), padding='same') for m in self.pooling_sizes]

  def call(self, x):
    
    x = self.input_conv(x) 

    if isinstance(self.pooling_sizes, int):
      x1 = self.MaxPooling2D(x)
      x2 = self.MaxPooling2D(x1)
      x3 = self.MaxPooling2D(x2)
      x_out = concatenate([x, x1,x2,x3], axis=-1)
    else:
      feats = [x1] + [max_pool(x) for max_pool in self.MaxPooling2D_list]
      x_out = concatenate(feats, axis=-1)

    out = self.output_conv(x_out) 
    return out

  def get_config(self):
    config = super(SPPBottleneck, self).get_config()
    config.update(
          {
            "filters": self.filters,
            "pool_sizes" : self.pooling_sizes,       
          }
    )
    return config 
  


############################################################################
#
#
############################################################################
@MODELS.register_module()
class CSPNeXt(BaseBackbone):
    #in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
            [256, 512, 6, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
            [256, 512, 6, True, False], [512, 768, 3, True, False],
            [768, 1024, 3, False, True]]
        }
  
    def __init__(self,      
            model_input_shape=(256,192),
            out_indices = (4, ),
            arch = 'P5',
            deepen_factor = 1.0,
            widen_factor = 1.0,
            expand_ratio = 0.5,
            data_preprocessor: dict = None, **kwargs):
        
        #super(CSPNeXt, self).__init__()
        self.arch_setting = self.arch_settings[arch]
        'input'
        self.model_input_shape = model_input_shape 
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.expand_ratio = expand_ratio

        super().__init__(input_size = (*model_input_shape,3),
                        data_preprocessor = data_preprocessor,
                        name = 'CSPNeXt')


    def forward(self, x):
    
        stem_channels = self.arch_setting[0][0]*self.widen_factor
        x = Conv2D_BN_SiLU(filters=stem_channels//2, kernel_size=(3,3), strides=(2,2), 
                use_bias=False, name='Stem-P0/Conv2D_BN_SiLU-1')(x)  #(b,128,96,32)

        x = Conv2D_BN_SiLU(filters=stem_channels//2, kernel_size=(3,3), strides=(1,1), 
                use_bias=False, name='Stem-P0/Conv2D_BN_SiLU-2')(x)  #(b,128,96,32) 
        
        x = Conv2D_BN_SiLU(filters=stem_channels, kernel_size=(3,3),strides=(1,1), 
                use_bias=False, name='Stem-P1/Conv2D_BN_SiLU')(x)    #(b,128,96,64)
        
        'P2->P5'
        for i, (in_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(self.arch_setting):
        
            out_channels =  out_channels*self.widen_factor
            x = Conv2D_BN_SiLU(filters=out_channels, kernel_size=(3,3), strides=(2,2), 
                        use_bias=False, name=f'Stage-{i+1}-P{i+2}/Conv2D_BN_SiLU')(x)       #(b,64,48,128)
        
            if use_spp : 
                x = SPPBottleneck(filters=out_channels, pool_sizes=5, name=f'Stage-{i+1}-P{i+2}/SPPBottleneck')(x)

            'CSPLayer'
            num_blocks = int( max(num_blocks*self.deepen_factor,1) )
            x = CSPLayer(filters=out_channels, expand_ratio=self.expand_ratio, num_blocks=num_blocks, 
                    add_identity=add_identity, use_depthwise=False, channel_attention=True,
                    name=f'Stage-{i+1}-P{i+2}/CSPLayer')(x)
        
        return x  
