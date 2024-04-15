from typing import Optional, Dict
from tensorflow.keras.layers import AveragePooling2D, Activation, Add
import tensorflow as tf
from lib.models.modules import BaseModule
from lib.layers import Conv2D_BN, attentions
#---------------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------------
class ShortCutModule(BaseModule):    
    VERSION = '1.0.0'

    r"""ShortCutModule used in ResBlocks
   
    Args:
        out_channels (int) : The output channels of this Module.
        strides (int):   stride of the block, defaults to 1. 
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        depoly (bool): determine depolyment config . default to None, 
                       depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfg depend on above args.
                       depoly = True => to use deployment config, only conv layer will be bulit
                       depoly = False => to use training config() , 
        name (str) : 'ResidualBlock'

    References:
        - [Based on implementation of 'BasicBlock' and 'Bottleneck' @mmpose] (https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/backbones/resnet.py)

    Note :
       - 

    Examples:
    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    x = Input(shape=(64,48,128))
    out = ShortCutModule(
                    out_channels = 256,
                    strides = 2,
                    avg_down = True,
                    bn_epsilon = 1e-5,
                    bn_momentum = 0.9,
                    deploy = False,
                    name  ='ShortCutModule'
    )(x)
    model = Model(x, out)
    model.summary(200)   

    """
    def __init__(self, 
                out_channels : int,
                strides : int = 1,
                avg_down : bool = False,
                **kwargs):
        super().__init__(**kwargs)

        'cfg for attention layers'
        self.out_channels = out_channels
        self.strides = strides
        self.avg_down = avg_down

  
    def build(self, input_shape):

        _, _, _, self.in_channels = input_shape
        'if out_channels=-1, out_channels=in_channels'
        if self.out_channels < 0: self.out_channels = self.in_channels

        'downsample (shorcut branch)'
        if self.strides != 1 or self.out_channels!=self.in_channels:
            if self.avg_down and self.strides != 1:
                self.downsample_ap = AveragePooling2D(
                    pool_size=self.strides, strides=self.strides, name = self.name+'AP'
                )
            
            self.downsample_conv = Conv2D_BN(
                    filters = self.out_channels,
                    kernel_size=1,
                    strides=  1 if self.avg_down else self.strides,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = None,
                    deploy=self.deploy,
                    name = self.name+'ConvBn'
            )

    def call(self, inputs):

        if hasattr(self, 'downsample_conv'):
            identity = self.downsample_conv(
                self.downsample_ap(inputs) if hasattr(self, 'downsample_ap') else inputs
            )
        else:
            identity = inputs 

        return  identity
    

#------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------
class BasicResModule(BaseModule):    
    VERSION = '1.0.0'
    r"""ResidualBlock  used in ResNet
   
    Args:
        out_channels (int) : The output channels of this Module.
        strides (int):   stride of the block, defaults to 1. 
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in DepthwiseConvModule and ConvModule, defaults to 'relu'.
        depoly (bool): determine depolyment config . default to None, 
                       depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfg depend on above args.
                       depoly = True => to use deployment config, only conv layer will be bulit
                       depoly = False => to use training config() , 
        name (str) : 'ResidualBlock'

    References:
        - [Based on implementation of 'BasicBlock' and 'Bottleneck' @mmpose] (https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/backbones/resnet.py)
        - [Inspired by  implementation of 'aot_block' @leondgarse] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/aotnet/aotnet.py)

    Note :
       - 

    Examples:
    ```python


    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    x = Input(shape=(64,48,128))

    out = BasicResModule(
                    out_channels = 256,
                    strides = 2,
                    avg_down  = True,
                    bn_epsilon = 1e-5,
                    bn_momentum = 0.9,
                    activation = 'relu',
                    psa_type = 'p',
                    se_ratio = 0.25,
                    use_eca = True,
                    deploy = False,
                    name  ='ResBottleneck'
    )(x)
    model = Model(x, out)
    model.summary(200)


    """
    def __init__(self, 
                out_channels : int,
                strides : int = 1,
                activation : str='relu',
                avg_down : bool = False,
                psa_type : Optional[str] = None,
                se_ratio : float = 0.,
                use_eca :  bool= False,
                **kwargs):
    
        super().__init__(activation=activation, **kwargs)
        if not (psa_type== None or isinstance(psa_type, str)):
            raise TypeError(
                    "the type of psa_type MUST be 'str' or None "
                    f", but got {type(psa_type)} @{self.__class__.__name__}"
            )
          
        'cfg for attention layers'
        self.psa_type = psa_type
        self.se_ratio = se_ratio
        self.use_eca = use_eca
        

        self.out_channels = out_channels
        self.strides = strides
        self.avg_down = avg_down

  
    def build(self, input_shape):

        _, _, _, self.in_channels = input_shape
        'if out_channels=-1, out_channels=in_channels'
        if self.out_channels < 0: self.out_channels = self.in_channels

        self.conv1 = Conv2D_BN(
                filters = self.out_channels,
                kernel_size = 3,
                strides=1,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                activation = self.act_name,
                deploy=self.deploy,
                name = self.name+f'ConvBn1'
        )

        self.conv2 =  Conv2D_BN(
                filters = self.out_channels,
                kernel_size = 3,
                strides=self.strides,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                activation = None,
                deploy=self.deploy,
                name = self.name+'ConvBn2'
        )

        'downsample (shorcut branch)'
        if self.strides != 1 or self.out_channels!=self.in_channels:
            self.downsample = ShortCutModule(
                    out_channels = self.out_channels,
                    strides = self.strides,
                    avg_down = self.avg_down,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    deploy = False,
                    name  = self.name+'Shortcut'
            )

        self.add = Add(name=self.name +'out_add')
        self.act  = Activation(
            self.act_name,
            name=self.name+f"out_{self.act_name}"
        )
        ''
        if self.se_ratio>0. : 
            self.se_layer = attentions.SqueezeAndExcitation(
                    se_ratio = self.se_ratio,
                    divisor = 8,
                    limit_round_down= 0.9,
                    use_bias  = False,
                    use_conv  = True,
                    hidden_activation = self.act_name,
                    output_activation = None,  
                    se_activation = 'sigmoid',  
                    name = self.name+"se"
            ) 
         
        if self.use_eca:
            self.eca_layer = attentions.EfficientChannelAttention(
                gamma = 2.0, 
                beta =1.0,
                name =self.name+'eca_layer'
            )

        if self.psa_type:  

            self.psa_layer = attentions.PolarizedSelfAttention(
                inplanes = -1, 
                mode = self.psa_type.lower(),                
                ln_epsilon = self.bn_epsilon,
                name = self.name+f"PSA{self.psa_type.lower()}"
            )
        
    def call(self, inputs):

        if hasattr(self, 'downsample'):
            identity = self.downsample(inputs)
        else:
            identity = inputs 
        

        x = self.conv1(inputs)
        
        if hasattr(self, 'psa_layer'):
            x = self.psa_layer(x)

        x = self.conv2(x)

        if hasattr(self,'se_layer'):
            x = self.se_layer(x)

        if hasattr(self,'eca_layer'):
            x = self.eca_layer(x)

        output = self.add([x,identity])
        return  self.act(output)
    

#------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------
class ResBottleneck (BaseModule):    
    VERSION = '1.0.0'
    ATTN_LAYERS = ['OutlookAttention', 'SimpleOutlookAttention', 'HaloAttention']

    r"""ResidualBlock  used in ResNet
   
    Args:
        out_channels (int) : The output channels of this Module.
        hidden_channel_ratio (int) :  stride of the block, defaults to 1. 
        strides (int):   stride of the block, defaults to 1. 
        avg_down (bool) :  Use AvgPool instead of stride conv when
            downsampling in the bottleneck
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in DepthwiseConvModule and ConvModule, defaults to 'relu'.
        depoly (bool): determine depolyment config . default to None, 
                       depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfg depend on above args.
                       depoly = True => to use deployment config, only conv layer will be bulit
                       depoly = False => to use training config() , 
        name (str) : 'ResidualBlock'

    References:
        - [Based on implementation of 'BasicBlock' and 'Bottleneck' @mmpose] (https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/backbones/resnet.py)
        - [Inspired by  implementation of 'aot_block' @leondgarse] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/aotnet/aotnet.py)

    Note :
       - 


    Examples:
    ```python

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    attn_layer_cfg = dict(
        type='SimpleOutlookAttention', 
        embed_dim = 64,
        num_heads  = 8,
        kernel_size  = 3, 
        attn_dropout_rate  =.0,
        name='OutlookAttnConv2'
    )
    x = Input(shape=(64,48,128))

    out = ResBottleneck(
            out_channels = 256,
            hidden_channel_ratio = 0.25,
            strides = 1,
            avg_down = False,
            bn_epsilon = 1e-5,
            bn_momentum = 0.9,
            activation = 'relu',
            deploy = False,
            attn_layer_cfg = attn_layer_cfg,
            se_ratio  = 0.25,
            use_eca  = True,
            name  ='ResBottleneck'
    )(x)
    model = Model(x, out)
    model.summary(200)

    print(model.get_layer('ResBottleneck_ConvBn1').weights[0][:,:,0,0])
    for layer in model.layers:
        print(layer.name)
        if hasattr(layer,'switch_to_deploy'):
            layer.switch_to_deploy()
    print(model.get_layer('ResBottleneck_ConvBn1').weights[0][:,:,0,0])
    model.summary(200)

    """
    def __init__(self, 
                out_channels : int,
                hidden_channel_ratio : float = 0.25, 
                strides : int = 1,
                activation : str='relu',
                avg_down : bool = False,
                attn_layer_cfg : Optional[Dict] = None,
                se_ratio : float = 0.,
                use_eca : bool = False,
                **kwargs):
    
        super().__init__(activation=activation, **kwargs)

        'verify cfg of attention layers'
        if not (attn_layer_cfg== None or isinstance(attn_layer_cfg, dict)):
            raise TypeError(
                    "the type of attn_layer_cfg MUST be 'dict' or None "
                    f", but got {type(attn_layer_cfg)} @{self.__class__.__name__}"
            )

        self.attn_layer_cfg = attn_layer_cfg
        self.se_ratio = se_ratio
        self.use_eca = use_eca
        
        self.hidden_channel_ratio = hidden_channel_ratio 
        self.out_channels = out_channels
        self.strides = strides
        self.attn_layer_cfg = attn_layer_cfg
        self.avg_down = avg_down
  
    def build(self, input_shape):

        _, _, _, self.in_channels = input_shape
        'if out_channels=-1, out_channels=in_channels'
        if self.out_channels < 0: self.out_channels = self.in_channels

        self.mid_channels = int( self.out_channels*self.hidden_channel_ratio)

        #self.branch_channels = ( self.in_channels*self.expand_times) //self.res_top_channels
        '1st Conv'
        self.conv1 = Conv2D_BN(
                filters = self.mid_channels,
                kernel_size = 1,
                strides=1,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                activation = self.act_name,
                deploy=self.deploy,
                name = self.name+f'ConvBn1'
        )
        '2nd Conv'
        if self.attn_layer_cfg is None :
            self.conv2 =  Conv2D_BN(
                    filters = self.mid_channels,
                    kernel_size=3,
                    strides=self.strides,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy=self.deploy,
                    name = self.name+'ConvBn2'
            )
        else:
            self.attn_type = self.attn_layer_cfg.pop('type')
            layer_name = self.attn_layer_cfg.pop('name')
            self.conv2 = getattr(
                attentions, self.attn_type
            )(**self.attn_layer_cfg, name=self.name+layer_name)

        '3rd Conv'
        self.conv3 = Conv2D_BN(
                filters = self.out_channels,
                kernel_size=1,
                strides=1,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                activation = None,
                deploy=self.deploy,
                name = self.name+f'ConvBn3'
        )

        'downsample (shorcut branch)'
        if self.strides != 1 or self.out_channels!=self.in_channels:
            self.downsample = ShortCutModule(
                    out_channels = self.out_channels,
                    strides = self.strides,
                    avg_down = self.avg_down,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    deploy = False,
                    name  = self.name+'Shortcut'
            )


        self.add = Add(name=self.name +'out_add')
        self.act  = Activation(
            self.act_name,
            name=self.name+f"out_{self.act_name}"
        )

        if self.se_ratio>0. : 
            self.se_layer = attentions.SqueezeAndExcitation(
                    se_ratio = self.se_ratio,
                    divisor = 8,
                    limit_round_down= 0.9,
                    use_bias  = False,
                    use_conv  = True,
                    hidden_activation = self.act_name,
                    output_activation = None,  
                    se_activation = 'sigmoid',  
                    name = self.name+f"se"
            )  

        if self.use_eca:
            self.eca_layer = attentions.EfficientChannelAttention(
                gamma = 2.0, 
                beta =1.0,
                name =self.name+'eca_layer'
            )        

    def call(self, inputs):

        if hasattr(self, 'downsample'):
            identity = self.downsample(inputs)
        else:
            identity = inputs 

        x = self.conv1(inputs)

        x = self.conv2(x)

        x = self.conv3(x) 

        'se attention layer'
        if hasattr(self, 'se_layer'):
            x = self.se_layer(x)
            
        'eca attention layer'
        if hasattr(self, 'eca_layer'):
            x = self.eca_layer(x)

        output = self.add([x,identity])
        return  self.act(output)