'tf layers'
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization,LayerNormalization, Activation, DepthwiseConv2D, Dense, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import ZeroPadding1D, ZeroPadding2D, Concatenate, Multiply, Reshape
from tensorflow import Tensor
import tensorflow as tf
from ..base_conv import Conv2D_BN


class ELAN(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r"""ElAN (Efficient Layer Aggregation Network )
    it was used in YoloV7'backbone  and its pafpn neck 

    Architecture :
        inputs = Input(shape=(80,80,256))
        elan_cfg = {
            in_planes = 128,
            out_channels = 512     
            stack_depth=6 , 
            stack_concats=[-1, -3, -5, -6],
            mid_ratio = 1.0
        }

        #[from, number, module, args]
        [-1, 1, Conv2D_BN, [128, 1, 1]],             #short_conv    (80,80,256)=>(80,80,128)
        [-2, 1, Conv2D_BN, [128, 1, 1]],             #prev_conv     (80,80,256)=>(80,80,128)
        [-1, 1, Conv2D_BN, [128, 3, 1]],             #conv_list[0]  (80,80,256)=>(80,80,128)
        [-1, 1, Conv2D_BN, [128, 3, 1]],             #conv_list[1]  (80,80,256)=>(80,80,128)
        [-1, 1, Conv2D_BN, [128, 3, 1]],             #conv_list[2]  
        [-1, 1, Conv2D_BN, [128, 3, 1]],             #conv_list[3]  
        [[-1, -3, -5, -6], 1, Concat, [axis=-1]],                    (80,80,128*4)
        [-1, 1, Conv, [512, 1, 1]],                  #out_conv       (80,80,128*4)=>(80,80,512)

    References:
            - [Graph of newwork ] (https://blog.csdn.net/pengxiang1998/article/details/128307956)
            - [Based on implementation of 'concat_stack'BrokenPipeError @leondgarse's keras_cv_attention_models  ] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/yolov7/yolov7.py)
    Args:
            in_planes (int) : filters of two stack_in_conv2D (1st and 2nd conv), defaults to None.
                              if in_planes = None,  it's a general condition that in_planes will be in_channels//2
            out_channels (int) : output channels of elan block, defaults to -1.
                                if out_channels=-1, elan output same channels as input tensor's channel.
            mid_ratio (float) : ratio of in_planes to hidden channels(filters of 3~6ith convs), defaults to 1. 
            stack_depth (int) : how many conv used in ELAN Block except final pw_conv, defaults to 6 . 
                                total conv_bn should be  stack_depth+1, here additional 1 means post_conv_bn to determine out_channels     
            stack_concats (int) : index of extracted features from each conv's outputs, defaults to [-1, -3, -5, -6].  
                                  if stack_concats is None, block will concat all feats from all conv_bn  
            bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
            bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
            activation (str) :  activation used in conv_bn blocks, defaults to 'relu'.
            name (str) : defaults to 'ELAN'.
    
    Note: 
        YoloV7Tiny -  { stack_depth=4 ,  mid_ratio=1.0, stack_concats= [-1, -2,-,3,-4] } in backbone
                      { stack_depth=4 ,  mid_ratio=1.0., stack_concats= [-1, -2,-,3,-4] } in pafpn (W_ELAN)

        YoloV7 -  { stack_depth=6 ,  mid_ratio=1.0, stack_concats= [-1, -3, -5, -6] } in backbone,
                  { stack_depth=6 ,  mid_ratio=0.5, stack_concats= [-1,-2,-3,-4, -5, -6] }  in pafpn (W_ELAN)


    Examples: 

    '''Python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    x = Input(shape=(256,256,128))
    out = ELAN(in_planes = 128, 
            out_channels = -1,
            mid_ratio = 1.0,
            depth = 6,
            concats = [-1, -3, -5, -6],
            activation = 'relu',
            name='ELAN')(x)
    model = Model(x, out)
    model.summary(150)

    """
    def __init__(self, 
                in_planes : Optional[int] = None, 
                out_channels :int = -1,
                mid_ratio : float= 1.0,
                stack_depth : int= 6,
                stack_concats : list = [-1, -3, -5, -6],
                bn_epsilon : float= 1e-5,
                bn_momentum : float= 0.9,
                activation = 'relu',
                deploy : Optional[bool] = None,
                name='ELAN',
                **kwargs)->None:
        super(ELAN, self).__init__(name=name)

        if stack_concats is not None and ( abs(min(stack_concats))>stack_depth) :
            raise ValueError(f"stack_concats = {stack_concats} out of bounds"
                             f"stack_depth = {stack_depth} @{self.__class__.__name__}"
            )
        
        if (not isinstance(bn_epsilon,(float))) or (not isinstance(bn_momentum,(float))):
            raise TypeError(f"bn_eps and  bn_momentum must be 'float' type @{self.__class__.__name__}"
                                f"but got eps:{type(bn_epsilon)}, momentum:{type(bn_momentum)}"
            )

        if not isinstance(activation,(str, type(None))):
            raise TypeError("activation must be 'str' type like 'relu'"
                         f"but got {type(activation)} @{self.__class__.__name__}"
            )
        self.deploy = deploy
        self.in_planes = in_planes
        self.out_channels = out_channels
        self.mid_ratio = mid_ratio
        self.depth = stack_depth
        self.concats_id = stack_concats if stack_concats is not None else [-(ii + 1) for ii in range(self.depth)] 
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum

        self.act_name = activation

    def build(self, input_shape):

        b,h,w,self.in_channels = input_shape

        if self.in_planes is None:
           self.in_planes =  self.in_channels //2

        if self.out_channels < 0: 
            self.out_channels =  self.in_channels 

        self.mid_filters = int(self.mid_ratio * self.in_planes)
        
        self.prev_conv = Conv2D_BN(filters = self.in_planes,
                                    kernel_size=1,
                                    strides=1,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum = self.bn_momentum,
                                    activation = self.act_name,
                                    deploy=self.deploy,
                                    name = f'1')
        
        self.short_conv = Conv2D_BN(filters = self.in_planes,
                                    kernel_size=1,
                                    strides=1,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum = self.bn_momentum,
                                    activation = self.act_name,
                                    deploy=self.deploy,
                                    name = f'2')
        
        self.conv_list = []
        for idx in range(self.depth - 2):
            self.conv_list.append( Conv2D_BN(filters = self.mid_filters,
                                            kernel_size=3,
                                            strides=1,
                                            bn_epsilon = self.bn_epsilon,
                                            bn_momentum = self.bn_momentum,
                                            activation = self.act_name,
                                            deploy=self.deploy,
                                            name = f"{idx+3}")
            )

        self.concat = Concatenate(axis=-1, name='concat')

        self.out_conv = Conv2D_BN(filters = self.out_channels,
                                    kernel_size=1,
                                    strides=1,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum = self.bn_momentum,
                                    activation = self.act_name,
                                    deploy=self.deploy,
                                    name = f"out")
    
    def call(self, x: Tensor)-> Tensor:

        main_branch = self.prev_conv(x)
        shortcut_branch = self.short_conv(x)
        'stack'
        feats = [shortcut_branch, main_branch]   
        for idx in range(self.depth - 2):
            main_branch  = self.conv_list[idx](feats[-1])
            feats.append(main_branch)

        gathered_fests = [feats[idx] for idx in self.concats_id]   
        nn = self.concat(gathered_fests)
        output = self.out_conv(nn)

        return output
    
    def get_config(self):
        config = super(ELAN, self).get_config()
        config.update(
                {
                "reparam_deploy"  : self.deploy, 
                "in_planes": self.in_planes,
                "out_channels": self.out_channels,
                "mid_ratio": self.mid_ratio,
                "mid_filters": self.mid_filters,
                "stack_depth": self.depth,
                "concats_id": self.concats_id,
                "bn_epsilon": self.bn_epsilon,
                "bn_momentum": self.bn_momentum,
                "act": self.act_name
                }
        )
        return config
    
    
    def switch_to_deploy(self):

        if self.deploy or self.deploy==None:
            return
        
        'get fused weight of conv_bn_1 and remove its bn layer'
        self.prev_conv.switch_to_deploy()
        prev_conv_weights  = self.prev_conv.weights
        self.short_conv.switch_to_deploy()
        short_conv_weights  = self.short_conv.weights

        bottleneck_weights = []
        for idx in range(self.depth - 2):
            self.conv_list[idx].switch_to_deploy()
            bottleneck_weights.append(self.conv_list[idx].weights)

        self.out_conv.switch_to_deploy()
        out_conv_weights  = self.out_conv.weights

        're build dw_conv_bn by seting input shape/ deploy = True / built = False'
        self.built = False
        self.deploy = True       
        super().__call__(tf.random.uniform(shape=(1,32,32,self.in_channels)))
        'update fused_weights to self.conv'
        self.prev_conv.set_weights(prev_conv_weights) 
        self.short_conv.set_weights(short_conv_weights)  

        for conv, weight in zip(self.conv_list, bottleneck_weights):
            conv.set_weights(weight)  

        self.out_conv.set_weights(out_conv_weights)  