from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

from tensorflow import Tensor
import tensorflow as tf
from tensorflow.keras.layers import Add, Concatenate
from lib.models.modules import BaseModule
from lib.layers import RepVGGConv2D as RepConvN
from lib.layers import Conv2D_BN,SeparableConv2D_BN



#------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------  
class RepNBottleneck(BaseModule):
    VERSION = '1.0.0'
    r""" RepNBottleneck


    """
    def __init__(
        self, 
        out_channels : int,
        exapnd_ratio : float = 0.5,
        kernel_sizes : Union[Tuple[int], List[int]] = [3,3],
        use_shortcut :bool=True,
        use_depthwise :bool=False,
        name : str ='RepNBottleneck',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.out_channels = out_channels
        self.exapnd_ratio = exapnd_ratio
        self.kernel_sizes = kernel_sizes
        self.use_shortcut = use_shortcut
        self.use_depthwise = use_depthwise
    

    def build(self,input_shape):
        _, _, _, self.in_channels = input_shape
        'if out_channels=-1, out_channels=in_channels'
        if self.out_channels < 0: self.out_channels = self.in_channels
        'verify whether apply short branch'
        self.use_shortcut = (self.use_shortcut and self.in_channels == self.out_channels)
        self.hidden_channels = int(self.out_channels*self.exapnd_ratio)


        self.conv1  = RepConvN(
            filters = self.hidden_channels,
            kernel_size=self.kernel_sizes[0],
            strides=1,
            groups=1,
            use_bias = False,
            use_depthwise = False,
            use_bn_identity  = False,
            activation = self.act_name,
            bn_epsilon = self.bn_epsilon,
            bn_momentum = self.bn_momentum,
            deploy = self.deploy,
            name = self.name +'Conv1'
        )
        if not self.use_depthwise :
            self.conv2  = Conv2D_BN(
                filters=self.out_channels,
                kernel_size=self.kernel_sizes[1],
                strides=1,
                activation = self.act_name,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                deploy = None,
                name = self.name +'Conv2'
            )
        else:
            self.conv2  = SeparableConv2D_BN(
                filters=self.out_channels,
                kernel_size=self.kernel_sizes[1],
                strides=1,
                activation = self.act_name,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                deploy = None,
                name = self.name +'Conv2'
            )  

        if self.use_shortcut :
            self.add = Add(name=self.name +'Add')   
                     
    def call(self, inputs : Tensor):
        deep = self.conv1(inputs)
        deep = self.conv2(deep)
        if self.use_shortcut :
            deep = self.add([deep,inputs]) 
    
        return deep
    
#------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------

class RepNCSP(BaseModule):
    VERSION = '1.0.0'
    R""" RepNCSP

    """
    def __init__(
        self, 
        out_channels : int,
        csp_depthes : int  = 1,
        exapnd_ratio : float = 0.5,
        kernel_sizes : Union[Tuple[int], List[int]] = [3,3],
        use_shortcut :bool=True,
        use_depthwise : bool = False,
        name : str ='RepNCSP',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_bottleneck = csp_depthes
        self.out_channels = out_channels
        self.exapnd_ratio = exapnd_ratio
        self.kernel_sizes = kernel_sizes
        self.use_shortcut = use_shortcut
        self.use_depthwise = use_depthwise
    
    def build(self,input_shape):
        _, _, _, self.in_channels = input_shape
        'if out_channels=-1, out_channels=in_channels'
        if self.out_channels < 0: self.out_channels = self.in_channels
        'verify whether apply short branch'
        self.hidden_channels = int(self.out_channels*self.exapnd_ratio)    

        self.conv1  = Conv2D_BN(
            filters=self.hidden_channels,
            kernel_size= 1,
            strides=1,
            activation = self.act_name,
            bn_epsilon = self.bn_epsilon,
            bn_momentum = self.bn_momentum,
            deploy = None,
            name = self.name +'Conv1'
        )

        self.conv2  = Conv2D_BN(
            filters=self.hidden_channels,
            kernel_size= 1,
            strides=1,
            activation = self.act_name,
            bn_epsilon = self.bn_epsilon,
            bn_momentum = self.bn_momentum,
            deploy = None,
            name = self.name +'Conv2'
        )     
        
        self.conv3  = Conv2D_BN(
            filters = self.out_channels,
            kernel_size = 1,
            strides = 1,
            activation = self.act_name,
            bn_epsilon = self.bn_epsilon,
            bn_momentum = self.bn_momentum,
            deploy = None,
            name = self.name +'Conv3'
        )  

        self.bottlenecks_list = []
        for idx in range(self.num_bottleneck):
            block_name = self.name+f'BottleNeck{idx+1}'
            setattr(self, block_name, 
                RepNBottleneck(  
                    out_channels = self.hidden_channels,
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

        self.concat = Concatenate(axis=-1, name=self.name+'Concat')

    def call(self, inputs : Tensor):

        
        deep = self.conv1(inputs)
        for rep_bottleneck in self.bottlenecks_list:
            deep = rep_bottleneck(deep)
        
        feats =self.concat( [ deep, self.conv2(inputs)] )
        return  self.conv3(feats)
    
#------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------
class RepNCSPELAN4(BaseModule):
    VERSION = "1.0.0"
    R"""" RepNCSPELAN4
    
    """
    def __init__(
        self, 
        out_channels : int,
        hidden_channels : int,
        csp_depthes : int  = 1,
        csp_exapnd_ratio : float = 0.5,
        kernel_sizes : Union[Tuple[int], List[int]] = [3,3],
        use_shortcut :bool=True,
        use_depthwise : bool = False,
        name : str ='RepNCSPELAN4',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.csp_depthes = csp_depthes
        self.csp_exapnd_ratio = csp_exapnd_ratio
        self.kernel_sizes = kernel_sizes
        self.use_shortcut = use_shortcut
        self.use_depthwise = use_depthwise

    def build(self,input_shape):
        _, _, _, self.in_channels = input_shape
        'if out_channels=-1, out_channels=in_channels'
        if self.out_channels < 0: self.out_channels = self.in_channels

        self.pre_conv  = Conv2D_BN(
                filters=self.hidden_channels,
                kernel_size=1,
                strides=1,
                activation = self.act_name,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                name = self.name +'PreConv'
        ) 

        self.split_layer = tf.keras.layers.Lambda(
            lambda x : tf.split(x,  num_or_size_splits=2, axis=-1), 
            name = self.name+'Split'
        )
        self.rep_csp_block1 = RepNCSP(
                out_channels = int(self.hidden_channels//2),
                csp_depthes = self.csp_depthes,
                exapnd_ratio  = self.csp_exapnd_ratio,
                kernel_sizes = self.kernel_sizes,
                use_shortcut = self.use_shortcut,
                use_depthwise = self.use_depthwise,
                activation = self.act_name,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                deploy = self.deploy,
                name  = self.name+ 'CSPBlock1'
        )
        self.trans_conv1  = Conv2D_BN(
                filters=int(self.hidden_channels//2),
                kernel_size=3,
                strides=1,
                activation = self.act_name,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                name = self.name +'TransConv1'
        ) 
        self.rep_csp_block2 = RepNCSP(
                out_channels = int(self.hidden_channels//2),
                csp_depthes = self.csp_depthes,
                exapnd_ratio  = self.csp_exapnd_ratio,
                kernel_sizes = self.kernel_sizes,
                use_shortcut = self.use_shortcut,
                use_depthwise = self.use_depthwise,
                activation = self.act_name,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                deploy = self.deploy,
                name  = self.name+ 'CSPBlock2'
        )
        self.trans_conv2  = Conv2D_BN(
                filters=int(self.hidden_channels//2),
                kernel_size=3,
                strides=1,
                activation = self.act_name,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                name = self.name +'TransConv2'
        ) 
        self.post_conv = Conv2D_BN(
                filters=self.out_channels,
                kernel_size=1,
                strides=1,
                activation = self.act_name,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                name = self.name +'PostConv'
        ) 
        
        self.concat = Concatenate(axis=-1, name=self.name+'Concat')


    def call(self, inputs : Tensor):

        deep = self.pre_conv(inputs)
        feats = self.split_layer(deep)  #feats= [short, deep]  
        feats.extend(
            conv ( rep_csp_block(feats[-1] )) 
            for rep_csp_block, conv in zip( [self.rep_csp_block1, self.rep_csp_block2],[self.trans_conv1, self.trans_conv2])
        )

        return  self.post_conv(self.concat(feats))    
#------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------