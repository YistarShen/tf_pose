from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from tensorflow.keras.layers import Concatenate, Add,  Activation
import tensorflow as tf
from lib.Registers import MODELS
from lib.models.modules import BaseModule
from lib.layers import Conv2D_BN



#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
@MODELS.register_module()
class ResidualStepsBlock (BaseModule):    
    VERSION = '1.0.0'
    r"""RSB  used in RSN 
   
    References:
            - [Based on implementation of ' RSB(BaseModule)' @mmpose] (https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/backbones/rsn.py)
            - [RSN paper ] (https://arxiv.org/pdf/2003.04030.pdf)


                            out_channels : int,

    Args:
        out_channels (int) : The output channels of this Module.
        expand_times (int) :  Times by which the in_channels are expanded, defaults to 26
        kernel_size (int) : The kernel size of step convolution layer (SeparableConv2D_BN), defaults to 3. 
        strides (int) :  stride of the block, defaults to 1. 
        res_top_channels(int) : Number of channels of feature output by ResNet_top. Defaul to 64, 
        num_steps (int): Numbers of steps in RSB, if steps=2, it means to split input to 2 branches
        use_depthwise (bool) : whether to use depthwise separable convolution for 1st convolution layer, defaults to false
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in DepthwiseConvModule and ConvModule, defaults to 'relu'.
        depoly (bool): determine depolyment config . default to None, 
                       depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfg depend on above args.
                       depoly = True => to use deployment config, only conv layer will be bulit
                       depoly = False => to use training config() , 
        name (str) : 'RSB'

    Note :
       - 


    Examples:
    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    x = Input(shape=(256,256,128))
    out = ResidualStepsBlock(out_channels = 256,
                expand_times = 26,
                num_steps = 4,
                strides = 2,
                res_top_channels = 64, 
                kernel_size = 3,
                use_depthwise = False,
                bn_epsilon = 1e-5,
                bn_momentum = 0.9,
                activation = 'silu',
                deploy = False,
                name  ='RSB'
    )(x)
    model = Model(x, out)

    print( model.get_layer('RSB_B1_conv1').weights[0][:,:,0,0])
    for layer in model.layers:
        if hasattr(layer,'switch_to_deploy'):
            for weight in layer.weights:
                print(weight.name)
            layer.switch_to_deploy()
            for weight in layer.weights:
                print(weight.name)
            print(layer.get_config())   
    print( model.get_layer('RSB_B1_conv1').weights[0][:,:,0,0])

    """
    def __init__(self, 
                out_channels : int,
                expand_times  : int = 26,
                num_steps : int = 4,
                strides : int = 1,
                res_top_channels : int = 64, 
                kernel_size : int = 3,
                use_depthwise : bool = False,
                activation : str='relu',
                name : str ='RSB',
                **kwargs):
    
        super(ResidualStepsBlock, self).__init__(name=name, 
                                                 activation=activation, 
                                                 **kwargs)
        
        if num_steps<=0:
            raise ValueError(f"num_steps must be > 0"
                             f"but got{num_steps} @{self.__class__.__name__}"
            )

        self.out_channels = out_channels
        self.num_steps = num_steps
        self.expand_times  = int(expand_times )
        self.res_top_channels = res_top_channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.use_depthwise = use_depthwise

        
    def build(self, input_shape):

        _, _, _, self.in_channels = input_shape
        'if out_channels=-1, out_channels=in_channels'
        if self.out_channels < 0: self.out_channels = self.in_channels

        self.branch_channels = ( self.in_channels*self.expand_times) //self.res_top_channels

        self.prev_conv = Conv2D_BN(
                    filters = self.branch_channels*self.num_steps,
                    kernel_size=1,
                    strides=self.strides,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy=self.deploy,
                    name = self.name+f'prev_pwconv'
        )
     
        if self.strides != 1 or self.out_channels!=self.in_channels:
            self.short_conv =  Conv2D_BN(
                    filters = self.out_channels,
                    kernel_size=1,
                    strides=self.strides,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = None,
                    deploy=self.deploy,
                    name = self.name+'short_pwconv'
            )

        self.branchs_conv_list = []
        for branch_idx in range(self.num_steps):
            branch_conv_list = []
            for i in range(branch_idx+1):
                name = self.name+f'B{branch_idx+1}_conv{i+1}'
                setattr(self, 
                    name, 
                    Conv2D_BN(filters = self.branch_channels,
                        kernel_size=self.kernel_size,
                        strides=1,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        activation = self.act_name,
                        deploy=self.deploy,
                        name = name
                    )
                )
                branch_conv_list.append(getattr(self, name))

            self.branchs_conv_list.append(branch_conv_list)

        self.concat = Concatenate(axis=-1, name=self.name+'branchs_concat')

        self.post_conv =  Conv2D_BN(
                    filters = self.out_channels ,
                    kernel_size=1,
                    strides=1,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = None,
                    deploy=self.deploy,
                    name = self.name+'post_pwconv'
        )
        
        self.add = Add(name=self.name +'out_add')
        self.rsn_act  = Activation(
            self.act_name,
            name=self.name+f"out_{self.act_name}"
        )

   
    def call(self, inputs):

        x = self.prev_conv(inputs)
        'identity'
        if hasattr(self, 'short_conv'):
            identity = self.short_conv(inputs)
        else:
            identity = inputs  
    
        'split'
        xs = tf.split(
            x, 
            num_or_size_splits=self.num_steps, 
            axis=-1, 
            name=self.name+'split'
        )
        step_deeps = []
        step_feats  = []
        for idx , x in enumerate(xs):
            ith_branch_conv_list = self.branchs_conv_list[idx]
            for ith , conv in enumerate(ith_branch_conv_list):
                if ith<idx:
                    x =  Add(
                        name=self.name +f'B{idx+1}_add{ith+1}'
                    )([x, step_feats[ith]])
                    x= conv(x)
                    step_feats[ith] = x
                else:
                    x= conv(x)
                    step_feats.append(x)
            step_deeps.append(x)
    
        output = self.concat(step_deeps)
        output = self.post_conv(output)
        output = self.add([output,identity])
        return  self.rsn_act(output)