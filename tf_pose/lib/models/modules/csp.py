

'CSP Layer Family'
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from tensorflow.keras.layers import Conv1D, Conv2D, DepthwiseConv2D, BatchNormalization, Dense
from tensorflow.keras.layers import ZeroPadding2D, Concatenate, Multiply, Reshape, Add, MaxPooling2D, GlobalAveragePooling2D, Activation
from tensorflow import Tensor
import tensorflow as tf
from lib.layers import Conv2D_BN, DepthwiseConv2D_BN, SeparableConv2D_BN
from lib.layers import ChannelAttention
from .base_module import BaseModule
#------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------
class CSPNeXtBottleneck(BaseModule):    
    VERSION = '1.0.0'
    r"""CSPNeXtBottleneck used in csp layer 
    CSPNeXt-Bottleneck 

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and ACT.
    The first convLayer has CONSTANT filter size of 3x3 and the second one has the
    filter size of 5x5.

    Architecture :

        in: (b,80,80,256) => CSPNeXtBottleneck => out: (b,80,80,256)
        GhostBottleNeck_cfg = {
                out_channels = 256,
                exapnd_ratio= 0.5,
                kernel_size= 5,
                use_shortcut =True,
                use_depthwise  = False
        }
        #[from, number, module, args]
        -------------------------------------------------------------------------------------
        [-1, 1, Conv2D_BN, [128, 3, 1]],                 # (80,80,256)=>(80,80,128) 
        [-1, 1, SeparableConv2D_BN, [256, 5, 1]],        # (80,80,128)=>(80,80,256)
        [-1, -2, Add, []],                               # (80,80,256)+(80,80,256)=>(80,80,256)
    

    References:
            - [Based on implementation of 'CSPNeXtBlock' @mmdet] (https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/layers/csp_layer.py#L50)
         
    Args:
        out_channels (int) : The output channels of this Module.
        exapnd_ratio (float) :  Ratio to adjust the number of channels of the hidden layer, defaults to 0.5.
                exapnd_ratio = hidden_channels//out_channels. 
        kernel_size (int) : The kernel size of the 2nd convolution layer (SeparableConv2D_BN), defaults to 5. 
        use_shortcut (bool) :  Whether to add identity to the out, defaults to True. 
                Only works when in_channels == out_channels. 
        use_depthwise (bool) : whether to use depthwise separable convolution for 1st convolution layer, defaults to false
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in DepthwiseConvModule and ConvModule, defaults to 'silu'.
        depoly (bool): determine depolyment config . default to None, 
                       depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfg depend on above args.
                       depoly = True => to use deployment config, only conv layer will be bulit
                       depoly = False => to use training config() , 
        name (str) : 'CSPNeXtBottleneck'

    Note :
       - it's very simillar darknet-bottleneck used in YoloV8,


    Examples:
    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    x = Input(shape=(256,256,3))
    out = CSPNeXtBottleneck(out_channels = 128,
                exapnd_ratio= 1.,
                kernel_size= 5,
                use_shortcut =True,
                use_depthwise  = True,
                bn_epsilon = 1e-5,
                bn_momentum = 0.9,
                activation = 'silu',
                deploy = False,
                name ='CSPNeXtBottleneck')(x)
    model = Model(x, out)

    print( model.get_layer('CSPNeXtBottleneck').weights[0][:,:,0,0])
    for layer in model.layers:
        if hasattr(layer,'switch_to_deploy'):
            for weight in layer.weights:
                print(weight.name)
            layer.switch_to_deploy()
            for weight in layer.weights:
                print(weight.name)
            print(layer.get_config())   
    print( model.get_layer('CSPNeXtBottleneck').weights[0][:,:,0,0])

    """
    def __init__(self, 
                out_channels : int,
                exapnd_ratio : float = 0.5,
                kernel_sizes : Union[Tuple[int], List[int]] = [3,5],
                use_shortcut :bool=True,
                use_depthwise : bool = False,
                name : str ='CSPNeXtBottleneck',
                **kwargs):
    
        super(CSPNeXtBottleneck, self).__init__(name=name, **kwargs)


        if (not isinstance(kernel_sizes,(Sequence)) or len(kernel_sizes)!=2 or not all(isinstance(x, int) for x in kernel_sizes)):
            raise ValueError("kernel_sizes must be tuple/list of 2 integers"
                             f"but got {kernel_sizes} @{self.__class__.__name__}"
            )        
        #self.deploy = deploy
        self.out_channels = out_channels
        self.exapnd_ratio = exapnd_ratio
        self.kernel_sizes = kernel_sizes
        self.use_shortcut = use_shortcut
        self.use_depthwise = use_depthwise

        
    def build(self, input_shape):

        _, _, _, self.in_channels = input_shape
        'if out_channels=-1, out_channels=in_channels'
        if self.out_channels < 0: self.out_channels = self.in_channels
        'verify whether apply short branch'
        self.use_shortcut = (self.use_shortcut and self.in_channels == self.out_channels)

        self.hidden_channels = int(self.out_channels*self.exapnd_ratio)

        'the 1st conv that can be SeparableConv2D or Conv2D'
        Conv = SeparableConv2D_BN if self.use_depthwise else Conv2D_BN
        self.conv1 = Conv(
                        filters=self.hidden_channels,
                        kernel_size=self.kernel_sizes[0],
                        strides=1,
                        activation = self.act_name,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        deploy = self.deploy,
                        name = self.name +'conv1')

        'the 2bd conv is SeparableConv2D'
        self.conv2 = SeparableConv2D_BN(
                        filters=self.out_channels,
                        kernel_size=self.kernel_sizes[1],
                        strides=1,
                        activation = self.act_name,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        deploy = self.deploy,
                        name = self.name +'conv2')  

        if self.use_shortcut :
            self.add = Add(name=self.name +'add')

        
    def call(self, inputs):
        deep = self.conv1(inputs)
        deep = self.conv2(deep)
        if self.use_shortcut :
            deep = self.add([deep,inputs]) 
        return deep   
    


#------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------
class DarkNetBottleneck(BaseModule):
    VERSION = '1.0.0'
    r"""CSPNeXtBottleneck used in csp layer
    DarkNet-Bottleneck was used in YoloV8

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has CONSTANT kernel size of 1x1 and the second one has the
    kernel size of 3x3.

    Architecture :
        in: (b,80,80,256) => DarkNetBottleneck => out: (b,80,80,256)
        DarkNetBottleneck = {
                out_channels = 256,
                exapnd_ratio= 0.5,
                kernel_size= 3,
                use_shortcut =True,
                use_depthwise  = True
        -------------------------------------------------------------------------------------
        #[from, number, module, args]
        [-1, 1, Conv2D_BN, [128, 1, 1]],                 # (80,80,256)=>(80,80,128) 
        [-1, 1, SeparableConv2D_BN, [256, 3, 1]],        # (80,80,128)=>(80,80,256)
        [-1, -2, Add, []],                               # (80,80,256)+(80,80,256)=>(80,80,256)
    

    References:
            - [Based on implementation of 'DarknetBottleneck' @mmdet] (https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/layers/csp_layer.py#L50)
            - [Based on implementation of 'Bottleneck' @ultralytics] (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py)
            - [inspired on implementation of 'CrossStagePartial' @keras-cv] (https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/backbones/csp_darknet/csp_darknet_utils.py#L190)
            - [Inspired on implementation of 'csp_with_2_conv' @leondgarse's] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/yolov8/yolov8.py)
    Args:
        out_channels (int) : The output channels of this Module.
        exapnd_ratio (float) : Ratio to adjust the number of channels of the hidden layer, defaults to 0.5.
                exapnd_ratio = hidden_channels//out_channels.
        kernel_sizes (int) : Tuple/list of 2 integers representing the sizes of the convolving kernel of 
                1st and 2nd Conv2D in each bottleneck,respectively.
                i.e. kernel_sizes = [kernel_szie_conv1, kernel_szie_conv2], defaults to [1,3].
        use_shortcut (bool) :  Whether to add identity to the out, defaults to True. 
                Only works when in_channels == out_channels. 
        use_depthwise (bool) : whether to use depthwise separable convolution for 2nd convolution layer, defaults to false
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in DepthwiseConvModule and ConvModule, defaults to 'swish'.
        depoly (bool): determine depolyment config . default to None, 
                depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfg depend on above args.
                depoly = True => to use deployment config, only conv layer will be bulit
                depoly = False => to use training config() , 
        name (str) : 'CSPNeXtBottleneck'

    Note :
       - it's very simillar darknet-bottleneck used in YoloV8,


    Examples:
    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    x = Input(shape=(256,256,3))
    out = DarkNetBottleneck(out_channels = 128,
                exapnd_ratio= 1.,
                kernel_size= 3,
                use_shortcut =True,
                use_depthwise  = True,
                bn_epsilon = 1e-5,
                bn_momentum = 0.9,
                activation = 'silu',
                deploy = False,
                name ='CSPNeXtBottleneck')(x)
    model = Model(x, out)

    print( model.get_layer('CSPNeXtBottleneck').weights[0][:,:,0,0])
    for layer in model.layers:
        if hasattr(layer,'switch_to_deploy'):
            for weight in layer.weights:
                print(weight.name)
            layer.switch_to_deploy()
            for weight in layer.weights:
                print(weight.name)
            print(layer.get_config())   
    print( model.get_layer('CSPNeXtBottleneck').weights[0][:,:,0,0])

    """
    def __init__(self, 
                out_channels : int,
                exapnd_ratio : float = 0.5,
                kernel_sizes : Union[Tuple[int], List[int]] = [1,3],
                use_shortcut :bool=True,
                use_depthwise : bool = False,
                name : str ='DarkNetBottleneck',
                **kwargs):
    
        super(DarkNetBottleneck, self).__init__(name=name, **kwargs)


        if (not isinstance(kernel_sizes,(Sequence)) or len(kernel_sizes)!=2 or not all(isinstance(x, int) for x in kernel_sizes)):
            raise ValueError("kernel_sizes must be tuple/list of 2 integers"
                             f"but got {kernel_sizes} @{self.__class__.__name__}"
            )   
        self.out_channels = out_channels
        self.exapnd_ratio = exapnd_ratio
        self.kernel_sizes = kernel_sizes
        self.use_shortcut = use_shortcut
        self.use_depthwise = use_depthwise

    def build(self, input_shape):

        _, _, _, self.in_channels = input_shape
        'if out_channels=-1, out_channels=in_channels'
        if self.out_channels < 0: self.out_channels = self.in_channels
        'verify whether to apply short branch'
        self.use_shortcut = (self.use_shortcut and self.in_channels == self.out_channels)

        self.hidden_channels = int(self.out_channels*self.exapnd_ratio)

        'the 2st conv is Conv2D'
        self.conv1 = Conv2D_BN(
                        filters=self.hidden_channels,
                        kernel_size=self.kernel_sizes[0],
                        strides=1,
                        activation = self.act_name,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        deploy = self.deploy,
                        name = self.name + 'conv1')

        'the 2nd conv that can be SeparableConv2D or Conv2D'
        Conv = SeparableConv2D_BN if self.use_depthwise else Conv2D_BN
        self.conv2 = Conv(
                        filters=self.out_channels,
                        kernel_size=self.kernel_sizes[1],
                        strides=1,
                        activation = self.act_name,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        deploy = self.deploy,
                        name = self.name + 'conv2')  

        if self.use_shortcut :
            self.add = Add(name=self.name + 'add')

        
    def call(self, inputs):
        deep = self.conv1(inputs)
        deep = self.conv2(deep)
        if self.use_shortcut :
            deep = self.add([deep,inputs]) 
        return deep 



#------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------   

class CrossStagePartial(BaseModule):
    VERSION = '1.0.0'
    r"""CrossStagePartial(C3) used in YoloV8 and CSPNeXt

    CrossStagePartial Module integrate 3 diiferent types that are 'C3', 'C2' and 'C2f'
    review https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py for more detail 
    we also can determine to use which bottleneck(cspnext_bottleneck and darknet_bottleneck) for each type
    

    CrossStagePartial_C3 @YoloX/CSPNeXt : Standard Implementation and contain 3 conv modules(main_conv/short_conv/final_conv) and Bottleneck blocks
    CrossStagePartial_C2 @xxxx : 2 conv modules(main_conv/final_conv) and Bottleneck blocks 
    CrossStagePartial_C2f @YoloV8: 2 conv modules(main_conv/final_conv) and Bottleneck blocks 

    Architecture :
        in: (b,80,80,256) => CrossStagePartial_C3 => out: (b,80,80,256)
        CrossStagePartial_C3 = {
                out_channels = 256,
                csp_type = 'C3',
                exapnd_ratio= 0.5,
                kernel_sizes = [3,5] ,
                use_cspnext_block = True,
                activation = 'swish'
                use_channel_attn = True,
        }


        -------------------------------------------------------------------------------------
        #[from, number, module, args]
        [-1, 1, Conv2D_BN, [128, 1, 1]],                 # (80,80,256)=>(80,80,128)    args ={filters, kernel_size, strides}
        [-2, 1, Conv2D_BN, [128, 1, 1]],                 # (80,80,256)=>(80,80,128) 
        [-1, 1, bottleneck, [128, [1,3], 1., True]],     # (80,80,128)=>(80,80,128)    args ={out_channels, kernel_sizes, exapnd_ratio, use_shortcut}
        [[-1, -3], 1, Concat, [-1]],                     # (80,80,128)x2 => (80,80,256)
        [-1, 1, ChannelAttnention, []],                  # (80,80,256) => (80,80,256)
        [-1, 1, Conv2D_BN, [256, 1, 1]],                 # (80,80,256)=>(80,80,256) 

    References:
            - [Based on implementation of 'CSPLayer' @mmdet] (https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/layers/csp_layer.py#L50)
            - [inspired on implementation of 'C3','C2','C2f' @ultralytics] (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py)
            - [inspired on implementation of 'CrossStagePartial' @keras-cv] (https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/backbones/csp_darknet/csp_darknet_utils.py#L190)
            - [inspired on implementation of 'csp_with_2_conv' @leondgarse] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/yolov8/yolov8.py)
    Args:
        out_channels (int) : The output channels of this Module.
        exapnd_ratio (float) : Ratio to adjust the number of channels of the hidden layer, defaults to 0.5.
                exapnd_ratio = hidden_channels//out_channels.
        kernel_sizes (int) :  Tuple/list of 2 integers representing the sizes of the convolving kernel of 
            1st and 2nd Conv2D in each bottleneck,respectively .i.e. [kernel_szie_conv1, kernel_szie_conv2]
            defaults to [1,3]@DarkNetBottleneck
        csp_type (int) : the type of CSP, defaults to 'C3'     
        csp_depthes (int) : the number of bottleneck blocks used in csp layer , defaults to 1. 
        use_shortcut (bool) : whether to use identity in bottleneck blocks, if True, the tensor before the bottleneck should be 
                added to the output of the bottleneck as a residual, defaults to True.
        use_depthwise (bool) : whether to use depthwise separable convolution for bottleneck blocks, defaults to false
        use_cspnext_block (bool) : Whether to use CSPNeXt block, default to False
        use_channel_attn (bool) : Whether to add channel attention in each stage, default to False
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in DepthwiseConvModule and ConvModule, defaults to 'swish'.
        depoly (bool): determine depolyment config . default to None, 
                depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfgs depend on above args.
                depoly = True => to use deployment config, only conv layer will be bulit
                depoly = False => to use training config() , 
        name (str) : 'CrossStagePartial_C3'

    Note :
       - use_cspnext_block=True and csp_type = 'C3':  kernel_sizes=[3,5], use_channel_attn=True, activation='swish', use_shortcut=True @CSPNeXt
       - use_cspnext_block=False and csp_type = 'C3': kernel_sizes=[1,3], use_channel_attn=False, activation='silu', use_shortcut=True @CSPDakNet
       - use_cspnext_block=False and csp_type = 'C2': kernel_sizes=[3,3], use_channel_attn=False, activation='silu', use_shortcut=True @CSPDakNet
       - use_cspnext_block=False and csp_type = 'C2f': kernel_sizes=[3,3], use_channel_attn=False, activation='silu', use_shortcut=True @CSPDakNet

    Examples:
    ```python

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    x = Input(shape=(256,256,128))

    #csp_type = 'C2f', use_cspnext_block = False @YoloV8
    out = CrossStagePartial(out_channels = 128,
                            exapnd_ratio= 0.5,
                            kernel_sizes = [3,3],
                            csp_type = 'C2f',
                            csp_depthes = 1,
                            use_shortcut = True,
                            use_cspnext_block = False,
                            use_depthwise  = False,
                            use_channel_attn = False,
                            bn_epsilon = 1e-5,
                            bn_momentum = 0.9,
                            activation = 'silu',
                            deploy = False,
                            name ='darknet_CSP_C2f')(x)  

    #csp_type = 'C2', use_cspnext_block = False
    out = CrossStagePartial(out_channels = 128,
                                exapnd_ratio= 0.5,
                                kernel_sizes = [3,3],
                                csp_type = 'C2',
                                csp_depthes = 1,
                                use_shortcut =True,
                                use_cspnext_block = False,
                                use_depthwise  = False,
                                use_channel_attn = False,
                                bn_epsilon = 1e-5,
                                bn_momentum = 0.9,
                                activation = 'silu',
                                deploy = False,
                                name ='darknet_CSP_C2')(x)  

    # csp_type = 'C3', use_cspnext_block = False
    out = CrossStagePartial(out_channels = 128,
                            exapnd_ratio= 0.5,
                            kernel_sizes = [1,3],
                            csp_type : str= 'C3',
                            csp_depthes = 2,
                            use_shortcut =True,
                            use_cspnext_block = False,
                            use_depthwise  = False,
                            use_channel_attn = False,
                            bn_epsilon = 1e-5,
                            bn_momentum = 0.9,
                            activation = 'silu',
                            deploy = False,
                            name ='darknet_CSP_C3_')(x)  

    # csp_type = 'C3', use_cspnext_block = True
    out = CrossStagePartial(out_channels = 128,
                                exapnd_ratio= 0.5,
                                kernel_sizes = [3,5],
                                csp_type : str= 'C3',
                                csp_depthes = 2,
                                use_shortcut =  False,
                                use_cspnext_block = True,
                                use_depthwise  = False,
                                use_channel_attn = True,
                                bn_epsilon = 1e-5,
                                bn_momentum = 0.9,
                                activation = 'swish',
                                deploy = False,
                                name ='cspnext_CSP_C3')     
    model = Model(x, out)
    model.summary(100)

    # switch_to_deploy
    print( model.get_layer('darknet_CSP_C3_').weights[0][:,:,0,0])
    for layer in model.layers:
        if hasattr(layer,'switch_to_deploy'):
            layer.switch_to_deploy()
            print('---------------------')
            for weight in layer.weights:
                print(weight.name)
        print(layer.get_config())   
    print( model.get_layer('darknet_CSP_C3_').weights[0][:,:,0,0])
    model.summary(100)          
    """
    def __init__(self, 
                out_channels : int,
                expand_ratio : float = 0.5,
                kernel_sizes : Optional[Union[Tuple[int], List[int]]] = [1,3],
                csp_depthes : int= 1,
                csp_type : str= 'C3',
                use_shortcut : bool = True,
                use_depthwise: bool = False,
                use_cspnext_block: bool = False,
                use_channel_attn : bool = False,
                name: str='CSP_C3',
                **kwargs):
        super(CrossStagePartial, self).__init__(name=name, **kwargs)

        if (not isinstance(kernel_sizes,(Sequence)) or len(kernel_sizes)!=2 or not all(isinstance(x, int) for x in kernel_sizes)):
            raise ValueError("kernel_sizes must be tuple/list of 2 integers"
                             f"but got {kernel_sizes} @{self.__class__.__name__}"
        )   
        if not isinstance(csp_type,str) or csp_type not in ['C3', 'C2', 'C2f']:
            raise ValueError("csp_type must be 'C3', 'C2' or  'C2f'\n"
                            f"but got csp_type={kernel_sizes} @{self.__class__.__name__}"
        )

        'basic module cfg'
        self.csp_type = csp_type
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.use_channel_attn = use_channel_attn
        'bottleneck cfg'
        self.kernel_sizes = kernel_sizes
        self.csp_depthes = csp_depthes
        self.use_shortcut =  use_shortcut
        self.use_cspnext_block = use_cspnext_block
        self.use_depthwise = use_depthwise

    def build(self, input_shape):
        _,_,_, self.in_channels = input_shape

        if self.out_channels < 0: self.out_channels = self.in_channels

        self.hidden_channels =  int(self.out_channels*self.expand_ratio)

        self.main_conv  = Conv2D_BN(filters = self.hidden_channels*(1 if self.csp_type=='C3' else 2) ,
                                kernel_size=1,
                                strides=1,
                                bn_epsilon = self.bn_epsilon,
                                bn_momentum = self.bn_momentum,
                                activation = self.act_name,
                                deploy  = self.deploy,
                                name = self.name+'main_conv')
        
        if self.csp_type=='C3' :
            self.short_conv = Conv2D_BN(filters = self.hidden_channels,
                                    kernel_size=1,
                                    strides=1,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum = self.bn_momentum,
                                    activation = self.act_name,
                                    deploy  = self.deploy,
                                    name = self.name+'short_conv')
        

        block = CSPNeXtBottleneck if self.use_cspnext_block else DarkNetBottleneck
        self.bottlenecks_list = []
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        for idx in range(self.csp_depthes):
            block_name = self.name+f'BottleNeckBlock_{idx+1}'
            setattr(self, block_name, 
                    block(  out_channels = self.hidden_channels,
                            kernel_sizes = self.kernel_sizes,
                            exapnd_ratio= 1.,
                            use_shortcut = self.use_shortcut,
                            use_depthwise = self.use_depthwise,
                            bn_epsilon = self.bn_epsilon,
                            bn_momentum = self.bn_momentum,
                            activation  = self.act_name,
                            deploy  = self.deploy,
                            name  = block_name
                    )
            )
            self.bottlenecks_list.append(getattr(self, block_name))
            '''
                CSPNeXtBottleneck(out_channels = 128,
                                exapnd_ratio= 1.,
                                kernel_size= 5,
                                use_shortcut =True,
                                use_depthwise  = True,
                                bn_epsilon = 1e-5,
                                bn_momentum = 0.9,
                                activation = 'silu',
                                deploy = False,
                                name ='CSPNeXtBottleneck')

                DarkNetBottleneck(out_channels = 128,
                                exapnd_ratio= 0.5,
                                kernel_size= 3,
                                use_shortcut =True,
                                use_depthwise  = False,
                                bn_epsilon = 1e-5,
                                bn_momentum = 0.9,
                                activation = 'swish',
                                deploy = False,
                                name ='DarkNetBottleneck')       
            '''
        self.concat = Concatenate(axis=-1, name=self.name+'concat')
        self.final_conv = Conv2D_BN(filters = self.out_channels,
                                kernel_size=1,
                                strides=1,
                                bn_epsilon = self.bn_epsilon,
                                bn_momentum = self.bn_momentum,
                                activation = self.act_name,
                                deploy  = self.deploy,
                                name = self.name+f"final_conv")
    
        if self.use_channel_attn:
            self.attn = ChannelAttention(use_bias = True,
                                        use_adpt_pooling = True,
                                        activation = 'hard_sigmoid',
                                        name=self.name+'attention')
            
    def call(self, x):

        deep = self.main_conv(x)

        if hasattr(self, 'short_conv'):
            short = self.short_conv(x)
        else:
            short, deep = tf.split(deep, num_or_size_splits=2, axis=-1, name=self.name+'split')

        if self.csp_type=='C2f': 
            feats = [short, deep] 
            for i in range(self.csp_depthes):
                deep = self.bottlenecks_list[i](feats[-1])
                feats.append(deep)
        else:
            for i in range(self.csp_depthes):
                deep = self.bottlenecks_list[i](deep)
            feats = [deep, short] 
        
        output = self.concat(feats)

        if self.use_channel_attn:
            output = self.attn(output)
        return self.final_conv(output)
        #return output
    
#------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------   
class CSPLayerWithTwoConv(BaseModule):
    VERSION = '1.0.0'
    r"""CrossStagePartial_C2f(CSPLayerWithTwoConv) used in YoloV8 
    Cross Stage Partial Layer with 2 convolutions.

    Architecture :
        in: (b,80,80,256) => CrossStagePartial_C3 => out: (b,80,80,256)
        CrossStagePartial_C2f = {
                out_channels = 256,
                exapnd_ratio= 0.5,
                csp_depthes = 2,
                kernel_sizes = [3,3] ,
                activation = 'swish'
        }

        -------------------------------------------------------------------------------------
        #[from, number, module, args]


    References:
            - [Based on implementation of 'CSPLayerWithTwoConv' @mmyolo] (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py)
            - [inspired on implementation of 'c2f' @ultralytics] (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py)
            - [inspired on implementation of 'csp_with_2_conv' @leondgarse] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/yolov8/yolov8.py)
    Args:
        out_channels (int) : The output channels of this Module.
        exapnd_ratio (float) : Ratio to adjust the number of channels of the hidden layer, defaults to 0.5.
                exapnd_ratio = hidden_channels//out_channels.
        kernel_sizes (int) :  Tuple/list of 2 integers representing the sizes of the convolving kernel of 
            1st and 2nd Conv2D in each bottleneck,respectively .i.e. [kernel_szie_conv1, kernel_szie_conv2]
            defaults to [1,3]@DarkNetBottleneck  
        csp_depthes (int) : the number of bottleneck blocks used in csp layer , defaults to 1. 
        use_shortcut (bool) : whether to use identity in bottleneck blocks, if True, the tensor before the bottleneck should be 
                added to the output of the bottleneck as a residual, defaults to True.
        use_depthwise (bool) : whether to use depthwise separable convolution for bottleneck blocks, defaults to false
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in DepthwiseConvModule and ConvModule, defaults to 'swish'.
        depoly (bool): determine depolyment config . default to None, 
                depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfgs depend on above args.
                depoly = True => to use deployment config, only conv layer will be bulit
                depoly = False => to use training config() , 
        name (str) : 'CrossStagePartial_C3'

    Note :
       - 

    Examples:
    ```python

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    x = Input(shape=(256,256,128))

    #csp_type = 'C2f', use_cspnext_block = False @YoloV8
    out = CrossStagePartial(out_channels = 128,
                            exapnd_ratio= 0.5,
                            kernel_sizes = [3,3],
                            csp_type = 'C2f',
                            csp_depthes = 1,
                            use_shortcut = True,
                            use_cspnext_block = False,
                            use_depthwise  = False,
                            use_channel_attn = False,
                            bn_epsilon = 1e-5,
                            bn_momentum = 0.9,
                            activation = 'silu',
                            deploy = False,
                            name ='darknet_CSP_C2f')(x)  

    model = Model(x, out)
    model.summary(100)

    # switch_to_deploy
    print( model.get_layer('darknet_CSP_C3_').weights[0][:,:,0,0])
    for layer in model.layers:
        if hasattr(layer,'switch_to_deploy'):
            layer.switch_to_deploy()
            print('---------------------')
            for weight in layer.weights:
                print(weight.name)
        print(layer.get_config())   
    print( model.get_layer('darknet_CSP_C3_').weights[0][:,:,0,0])
    model.summary(100)          
    """
    def __init__(self, 
                out_channels : int,
                expand_ratio : float = 0.5,
                kernel_sizes : Optional[Union[Tuple[int], List[int]]] = [3,3],
                csp_depthes : int= 1,
                use_shortcut : bool = True,
                use_depthwise: bool = False,
                name: str='CSP_C2f',
                **kwargs):
        super(CSPLayerWithTwoConv, self).__init__(name=name,  **kwargs)


        if (not isinstance(kernel_sizes,(Sequence)) or len(kernel_sizes)!=2 or not all(isinstance(x, int) for x in kernel_sizes)):
            raise ValueError("kernel_sizes must be tuple/list of 2 integers"
                             f"but got {kernel_sizes} @{self.__class__.__name__}"
        )   

        'basic module cfg'
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        'bottleneck cfg'
        self.kernel_sizes = kernel_sizes
        self.csp_depthes = csp_depthes
        self.use_shortcut =  use_shortcut
        self.use_depthwise = use_depthwise

        
    def build(self, input_shape):
        _,_,_, self.in_channels = input_shape

        if self.out_channels < 0: self.out_channels = self.in_channels

        self.hidden_channels =  int(self.out_channels*self.expand_ratio)

        self.main_conv  = Conv2D_BN(filters = self.hidden_channels*2,
                                kernel_size=1,
                                strides=1,
                                bn_epsilon = self.bn_epsilon,
                                bn_momentum = self.bn_momentum,
                                activation = self.act_name,
                                deploy  = self.deploy,
                                name = self.name+'main_conv')
        

        self.bottlenecks_list = []
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        for idx in range(self.csp_depthes):
            block_name = self.name+f'BottleNeckBlock_{idx+1}'
            setattr(self, block_name, 
                    DarkNetBottleneck( out_channels = self.hidden_channels,
                                    kernel_sizes = self.kernel_sizes,
                                    exapnd_ratio= 1.,
                                    use_shortcut = self.use_shortcut,
                                    use_depthwise = self.use_depthwise,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum = self.bn_momentum,
                                    activation  = self.act_name,
                                    deploy  = self.deploy,
                                    name  = block_name
                    )
            )
            self.bottlenecks_list.append(getattr(self, block_name))

        self.concat = Concatenate(axis=-1, name=self.name+'concat')
        self.final_conv = Conv2D_BN(filters = self.out_channels,
                                kernel_size=1,
                                strides=1,
                                bn_epsilon = self.bn_epsilon,
                                bn_momentum = self.bn_momentum,
                                activation = self.act_name,
                                deploy  = self.deploy,
                                name =self.name+ f"final_conv")
    
    def call(self, x : Tensor) ->Tensor:
        deep = self.main_conv(x)
        short, deep = tf.split(deep, num_or_size_splits=2, axis=-1, name=self.name+'split')
        feats = [short, deep] 
        for i in range(self.csp_depthes):
            deep = self.bottlenecks_list[i](feats[-1])
            feats.append(deep)
        output = self.concat(feats)
        return self.final_conv(output)
    


#------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------  
class CrossStagePartial_C1(BaseModule):
    VERSION = '1.0.0'
    r"""CrossStagePartial_C1(csp_c1) 
    Simple version, CSP Bottleneck with 1 convolution

    References:
            - [Based on @ultralytics's C1(nn.Module) torch implementation  (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py)
           
     Args:
            out_channels (int) : The output channels of this Module.
            csp_depthes (int) : the number of bottleneck blocks used in csp layer , defaults to 1. 
            use_depthwise (bool) : whether to use depthwise separable convolution for bottleneck blocks, defaults to false
            bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
            bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
            activation (str) : activation used in DepthwiseConvModule and ConvModule, defaults to 'swish'.
            depoly (bool): determine depolyment config . default to None, 
                    depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfgs depend on above args.
                    depoly = True => to use deployment config, only conv layer will be bulit
                    depoly = False => to use training config() , 
            name (str) :'CSP_C1'

    Examples:
    ```python

                
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        x = Input(shape=(256,256,128))

        out = CrossStagePartial_C1(out_channels = 128,
                                    csp_depthes = 2,
                                    use_depthwise  = False,
                                    bn_epsilon = 1e-5,
                                    bn_momentum = 0.9,
                                    activation = 'silu',
                                    deploy = False,
                                    name ='CSP_C1')(x)  
        model = Model(x, out)
        model.summary(100)
        print( model.get_layer('CSP_C1').weights[0][:,:,0,0])
        for layer in model.layers:
            if hasattr(layer,'switch_to_deploy'):
                layer.switch_to_deploy()
                print('---------------------')
                for weight in layer.weights:
                    print(weight.name)
            print(layer.get_config())   

        print( model.get_layer('CSP_C1').weights[0][:,:,0,0])
        model.summary(100)  
    """
    def __init__(self, 
                out_channels : int,
                csp_depthes : int= 2,
                use_depthwise: bool = False,
                name: str='CSP_C1',
                **kwargs):
        super(CrossStagePartial_C1, self).__init__(name=name, **kwargs)

        self.out_channels = out_channels
        self.csp_depthes = csp_depthes
        self.use_depthwise = use_depthwise

    def build(self, input_shape):
        _,_,_, self.in_channels = input_shape
        self.conv1_bn = Conv2D_BN(filters = self.out_channels,
                                kernel_size=1,
                                strides=1,
                                bn_epsilon = self.bn_epsilon,
                                bn_momentum = self.bn_momentum,
                                activation = self.act_name,
                                deploy  = self.deploy,
                                name = self.name + 'conv1_bn')
        name = self.name +'BottleNeckBlock_'
        self.bottlenecks = tf.keras.Sequential(
                        [ Conv2D_BN(filters = self.out_channels,
                                   kernel_size=3,
                                    strides=1,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum = self.bn_momentum,
                                    activation = self.act_name,
                                    deploy  = self.deploy,
                                    name =  name+f'{idx+1}') 
                            if self.use_depthwise 
                            else 
                            DepthwiseConv2D_BN(kernel_size=3,
                                            strides=1,
                                            bn_epsilon = self.bn_epsilon,
                                            bn_momentum = self.bn_momentum,
                                            activation = self.act_name,
                                            deploy = self.deploy,  
                                            name =  name+f'{idx+1}')
                            for idx in range(self.csp_depthes)], name=self.name+'bottleneck_seq')
   
        self.add = Add(name=self.name + 'add')

    def call(self, x):
        deep = self.conv1_bn(x)
        short = deep 
        deep = self.bottlenecks(deep)
        output = self.add([deep, short])
        return output

    # def switch_to_deploy(self):

    #     if self.deploy or self.deploy==None:
    #         return
        
    #     'get fused weight of conv_modules and remove their bn layer'
    #     self.conv1_bn.switch_to_deploy()
    #     conv1_bn_weights  = self.conv1_bn.weights

    #     bottleneck_weights = []
    #     for layer in self.bottlenecks.layers:
    #         layer.switch_to_deploy()
    #         bottleneck_weights.append(layer.weights)

    #     'rebuild block by seting a input shape/ deploy=True / built=False'
    #     self.built = False
    #     self.deploy = True       
    #     super().__call__(tf.random.uniform(shape=(1,32,32,self.in_channels)))

    #     'update fused_weights to all conv_modules'
    #     self.conv1_bn.set_weights(conv1_bn_weights) 
    #     for layer, weight in zip(self.bottlenecks.layers, bottleneck_weights):
    #         layer.set_weights(weight)   
    # 
#------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------    