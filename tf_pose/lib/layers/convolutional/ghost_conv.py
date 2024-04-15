
import math
import tensorflow as tf
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from lib.layers import Conv2D_BN, DepthwiseConv2D_BN
from tensorflow.keras.layers import AveragePooling2D, Concatenate, Multiply, Resizing
#--------------------------------------------------------------------------------------------------------------
#
#--------------------------------------------------------------------------------------------------------------
class GhostConv2D(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r""" GhostConv2D (ghost_module)

    A Light-weight convolutional neural networks (CNNs) are specially designed for
    applications on mobile devices with faster inference speed.
    decoupled fully connected attention (dfc) used in moddule is a hardware-friendly attention mechanism (
    it was appiled on GhostNetV2)

    GhostConv2D was used in GhostNetV2/GhostNetV1/GhostBottoleNeck

    Return :
           tf.Tensor

    Architecture :
        in: (b,40,40,128) => GhostBottleNeck => out: (b,40,40,256)

        GhostBottleNeck_cfg = {
            filters = 256,
            kernel_size = 1, 
            dw_kernel_size =3,
            strides = 1,    
            use_depthwise= True,
            use_dfc_attention = False,
        }



        #[from, number, module, args]
        [-1, 1, Conv, [128, 1, 1]]              #arbitrary dummy input with shape (80,80,128)
        -------------------------------------------------------------------------------------
        [-1, 1, dfc_block, [256, 3, 1]],        # (80,80,128)=>(80,80,256) 
        [-2, 1, Conv2D_BN, [128, 1, 1]],        # (80,80,128)=>(80,80,128)
        [-1, 1, DWConv2D_BN, [128, 3, 1]],      # (80,80,128)=>(80,80,128)
        [[-1, -2,], 1, Concat, [axis=-1]],      # (80,80,128*2)
        [-1, -4, MUL, []],                      # (80,80,256)*(80,80,256)=>(80,80,256)


    Note : 
        - this conv module was implemented by Conv2D_BN and DepthwiseConv2D_BN
        - support fuse conv+bn as one conv layer by switch_to_deploy, but need some experiments to verify
        - although dfc attention can support strides=2, strides should be always 1 in gerneral (use another conv with stride=2 to downsample in ghost-bottoleneck). 
    

    References:
        - [GhostNetV2  paper] (https://arxiv.org/pdf/2211.12905.pdf)
        - [Based on GhostNetV2 implement @likyoo] (https://github.com/likyoo/GhostNetV2-PyTorch/blob/main/ghostnetv2.py)
        - [Inspired by @ultralytics's GhostConv torch module] (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py)
        - [Inspired by @leondgarse's ghost_module] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/ghostnet/ghostnet_v2.py)
    
    Args:
        filters (int): out_channels.
        kernel_size (int) :  kernel of primary conv, defaults to 1.
        dw_kernel_size (int) :  kernel of cheap conv, defaults to 2
        strides (int) : stride of primary conv. defaults to 1. 
                        in general, strides should be always is 1 . 
        use_depthwise (bool) : whether use DepthwiseConv2D for cheap conv , default to True.
                               if not, cheap_conv will apply Conv2D(groups=in_channels),
        use_dfc_attention (bool) : whether use dfc attention block for residual branch, default to False
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) = activation used in conv_bn blocks, defaults to None.
        depoly (bool): determine depolyment config . default to None, 
                       depoly = None => diable re-parameterize attribute (turn off switch_to_deploy), all cfg depend on above args.
                       depoly = True => to use deployment config, only conv layer will be bulit
                       depoly = False => to use training config() , 
                                if depoly=False, switch_to_deploy can convert trained weights(conv_kernel+conv_bias+bn) to deployed weights (only conv_kernel)
        name (str) :'GhostConv2D'
    
    Note:

    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    x = Input(shape=(256,256,16))
    out = GhostConv2D(filters = 16,
                    kernel_size =1,
                    dw_kernel_size =3,
                    strides=1,
                    use_depthwise = True,
                    use_dfc_attention = False,
                    bn_epsilon = 1e-5,
                    bn_momentum = 0.9,
                    activation = 'relu',
                    deploy = False,
                    name = 'GhostConv2D')(x)
    model = Model(x, out)
    model.summary(100)


    """
    def __init__(self,
            filters : int,
            kernel_size : int=1,
            dw_kernel_size : int=3,
            strides: int =1,
            use_depthwise : bool = True,
            use_dfc_attention : bool = False,
            bn_epsilon : float= 1e-5,
            bn_momentum : float= 0.9,
            activation : Optional[str] = 'relu',
            deploy : Optional[bool] = None,
            name : str='Conv2D', **kwargs):
        super().__init__(name=name,  **kwargs)
      
        if isinstance(kernel_size, (list, tuple)) :
            self.kernel_size = kernel_size
        else :
            self.kernel_size = (kernel_size, kernel_size)

        if isinstance(dw_kernel_size, (list, tuple)) :
            self.dw_kernel_size = dw_kernel_size
        else :
            self.dw_kernel_size = (dw_kernel_size, dw_kernel_size)

        if not isinstance(activation,(str, type(None))):
            raise TypeError("activation must be 'str' type like 'relu'"
                        f"but got {type(activation)} ")

        self.deploy = deploy
        self.filters = filters
        self.strides = strides
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.use_depthwise = use_depthwise
        self.use_dfc_attention = use_dfc_attention

        self.act_name = activation
        self.deploy = deploy


    def build(self, input_shape):

        _, h, w, self.in_channels = input_shape
        if self.filters < 0:
            self.filters =  self.in_channels

        self.hidden_channels = int(math.ceil(float(self.filters) / 2.))
        #self.hidden_channels = self.filters//2

        self.primary_conv = Conv2D_BN(filters = self.hidden_channels,
                                        kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        bn_epsilon = self.bn_epsilon,
                                        bn_momentum = self.bn_momentum,
                                        activation = self.act_name,
                                        deploy = self.deploy,
                                        name = f'primary_conv')
        
        if self.use_depthwise:
            self.cheap_conv = DepthwiseConv2D_BN(
                                        kernel_size=self.dw_kernel_size,
                                        strides=1,
                                        bn_epsilon = self.bn_epsilon,
                                        bn_momentum = self.bn_momentum,
                                        activation = self.act_name,
                                        deploy = self.deploy,
                                        name = 'cheap_conv')
        else:
            self.cheap_conv = Conv2D_BN(filters = self.hidden_channels,
                                        kernel_size=self.dw_kernel_size,
                                        strides=1,
                                        groups = self.hidden_channels,
                                        bn_epsilon = self.bn_epsilon,
                                        bn_momentum = self.bn_momentum,
                                        activation = self.act_name,
                                        deploy = self.deploy,
                                        name = f'cheap_conv')   
        
        self.concat = Concatenate(axis=-1, name='concat')

        if self.use_dfc_attention :
            ' decoupled_fully_connected_attention used in ghostnetv2'
            self.dfc_attn = tf.keras.Sequential(
                [
                    AveragePooling2D(pool_size=(2, 2), 
                                    strides=2, 
                                    padding='valid', 
                                    name='attn_AP'
                    ),
                    Conv2D_BN(filters = self.filters,
                            kernel_size = self.kernel_size,
                            strides=self.strides,
                            bn_epsilon = self.bn_epsilon,
                            bn_momentum = self.bn_momentum,
                            name = f'attn_1'
                    ),
                    DepthwiseConv2D_BN(kernel_size=(1,5),
                                        bn_epsilon = self.bn_epsilon,
                                        bn_momentum = self.bn_momentum,
                                        name = 'attn_2'
                    ),
                    DepthwiseConv2D_BN(kernel_size=(5,1),
                                        bn_epsilon = self.bn_epsilon,
                                        bn_momentum = self.bn_momentum,
                                        name = 'attn_3'
                    ),
                    tf.keras.layers.Lambda(
                        lambda x: tf.keras.activations.sigmoid(x)
                    ),
                    Resizing( h//self.strides, w//self.strides, interpolation="nearest", name='attn_resize')

                ],
                name="dfc_attn_block",
            )
            self.mul = Multiply(name='mul')

    def call(self,inputs) ->tf.Tensor:
        'GhostModule_v1'
        x = self.primary_conv(inputs)
        ghost_out = self.concat([x, self.cheap_conv(x)])
        'GhostModule_v2 (add dfc_attention)'
        if self.use_dfc_attention:
           #ghost_res = self.dfc_attn(inputs)
           ghost_out =  self.mul([ghost_out, self.dfc_attn(inputs)])
        
        return ghost_out

    def get_config(self):
        config = super().get_config()
        config.update(
                {
                    "filters": self.filters,
                    "kernel_size": self.kernel_size,
                    "dw_kernel_size": self.dw_kernel_size,
                    "strides": self.strides,
                    "use_dfc_attention" : self.use_dfc_attention,
                    "use_depthwise" : self.use_depthwise,   
                    "bn_epsilon" : self.bn_epsilon,
                    "bn_momentum" : self.bn_momentum,
                    "reparam_deploy" : self.deploy,
                    "activation" : self.act_name,
                    "params": super().count_params()
                }
        )
        return config
    
    def switch_to_deploy(self):

        if self.deploy or self.deploy==None:
            return
        
        'get fused weight of conv_bn_1 and remove its bn layer'
        self.primary_conv.switch_to_deploy()
        primary_conv_weights  = self.primary_conv.weights
        self.cheap_conv.switch_to_deploy()
        cheap_conv_weights  = self.cheap_conv.weights
        're build dw_conv_bn by seting input shape/ deploy = True / built = False'
        self.built = False
        self.deploy = True 
        super().__call__(tf.random.uniform(shape=(1,32,32,self.in_channels)))
        'update fused_weights to self.conv'
        self.primary_conv.set_weights(primary_conv_weights) 
        self.cheap_conv.set_weights(cheap_conv_weights)    