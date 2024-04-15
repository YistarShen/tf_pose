# 'tf layers'
# from typing import Any, Callable, Dict, List, Optional, Tuple, Union
# from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization,LayerNormalization, Activation, DepthwiseConv2D, Dense, MaxPooling2D, GlobalAveragePooling2D
# from tensorflow.keras.layers import ZeroPadding1D, ZeroPadding2D, Concatenate, Multiply, Reshape
# from tensorflow import Tensor
# import tensorflow as tf
# import numpy as np
# from ..base_conv import Conv2D_BN

# #-------------------------------------------------------------------------------------------
# #
# #-------------------------------------------------------------------------------------------
# class ELAN(tf.keras.layers.Layer):
#     VERSION = '1.0.0'
#     r"""ELAN_Layer()

#     """
#     def __init__(self, 
#                 in_planes : int, 
#                 out_channels :int = -1,
#                 mid_ratio : int= 1.0,
#                 depth : int= 6,
#                 concats : list = [-1, -3, -5, -6],
#                 bn_epsilon : float= 1e-5,
#                 bn_momentum : float= 0.9,
#                 activation = 'relu',
#                 name='ELAN',
#                 **kwargs):
#         super(ELAN, self).__init__(name=name)

#         if (not isinstance(bn_epsilon,(float))) or (not isinstance(bn_momentum,(float))):
#                 raise TypeError(f"bn_eps and  bn_momentum must be 'float' type @{self.__class__.__name__}"
#                                 f"but got eps:{type(bn_epsilon)}, momentum:{type(bn_momentum)}"
#                 )

#         if not isinstance(activation,(str, type(None))):
#             raise TypeError("activation must be 'str' type like 'relu'"
#                          f"but got {type(activation)} @{self.__class__.__name__}"
#             )

#         self.in_planes = in_planes
#         self.out_channels = out_channels
#         self.mid_ratio = mid_ratio
#         self.mid_filters = int(self.mid_ratio * self.in_planes)
#         self.depth = depth
#         self.concats_id = concats if concats is not None else [-(ii + 1) for ii in range(self.depth)] 
#         self.bn_epsilon = bn_epsilon
#         self.bn_momentum = bn_momentum

#         self.act = activation

#     def build(self, input_shape):
#         b,h,w,c = input_shape
#         if self.out_channels < 0: 
#             self.out_channels =  c
    
#         self.prev_conv = Conv2D_BN(filters = self.in_planes,
#                                     kernel_size=1,
#                                     strides=1,
#                                     bn_epsilon = self.bn_epsilon,
#                                     bn_momentum = self.bn_momentum,
#                                     activation = self.act,
#                                     name = f'1')
        
#         self.short_conv = Conv2D_BN(filters = self.in_planes,
#                                     kernel_size=1,
#                                     strides=1,
#                                     bn_epsilon = self.bn_epsilon,
#                                     bn_momentum = self.bn_momentum,
#                                     activation = self.act,
#                                     name = f'2')
        
#         self.conv_list = []
#         for idx in range(self.depth - 2):
#             self.conv_list.append( Conv2D_BN(filters = self.mid_filters,
#                                             kernel_size=3,
#                                             strides=1,
#                                             bn_epsilon = self.bn_epsilon,
#                                             bn_momentum = self.bn_momentum,
#                                             activation = self.act,
#                                             name = f"{idx+3}")
#             )

#         self.concat = Concatenate(axis=-1, name=self.name + 'concat')

#         self.out_conv = Conv2D_BN(filters = self.out_channels,
#                                     kernel_size=1,
#                                     strides=1,
#                                     bn_epsilon = self.bn_epsilon,
#                                     bn_momentum = self.bn_momentum,
#                                     activation = self.act,
#                                     name = f"out")
    
#     def call(self, x):

#         main_branch = self.prev_conv(x)
#         shortcut_branch = self.short_conv(x)
#         'stack'
#         feats = [shortcut_branch, main_branch]   
#         for idx in range(self.depth - 2):
#             main_branch  = self.conv_list[idx](feats[-1])
#             feats.append(main_branch)

#         gathered_fests = [feats[idx] for idx in self.concats_id]   
#         nn = self.concat(gathered_fests)
#         output = self.out_conv(nn)

#         return output
    
#     def get_config(self):
#         config = super(ELAN, self).get_config()
#         config.update(
#                 {
#                 "in_planes": self.in_planes,
#                 "out_channels": self.out_channels,
#                 "mid_ratio": self.mid_ratio,
#                 "mid_filters": self.mid_filters,
#                 "depth": self.depth,
#                 "concats_id": self.concats_id,
#                 "bn_epsilon": self.bn_epsilon,
#                 "bn_momentum": self.bn_momentum,
#                 "act": self.act
#                 }
#         )
#         return config
    
# #-------------------------------------------------------------------------------------------
# #
# #-------------------------------------------------------------------------------------------
# class MPConvBlock(tf.keras.layers.Layer):
#     r"""MaxPoolAndStrideConvBlock(csp_downsample)
#     this block used in YoloV7 for downsample

#     Reference:
#         - [YoloV7 Intro](https://medium.com/@nahidalam/understanding-yolov7-neural-network-343889e32e4e)

#     """
#     def __init__(self, 
#                 out_channels :int,
#                 kernel_size : int = 2, 
#                 strides : int = 2,
#                 bn_epsilon : float= 1e-5,
#                 bn_momentum : float= 0.9,
#                 activation = 'relu',
#                 name='MP_Conv2D', **kwargs):
#         super(MPConvBlock, self).__init__(name=name)
    
#         if isinstance(kernel_size, (list, tuple)) :
#             self.kernel_size = kernel_size
#         else :
#             self.kernel_size = (kernel_size, kernel_size)

#         self.strides = (strides,strides)

#         self.filters = (out_channels//2 )
#         self.bn_epsilon = bn_epsilon
#         self.bn_momentum = bn_momentum

        
#         if not isinstance(activation,(str, type(None))):
#             raise TypeError("activation must be 'str' type like 'relu'"
#                          f"but got {type(activation)} @{self.__class__.__name__}")
        
#         self.act = activation


#     def build(self, input_shape):

#         self.B1_mp = MaxPooling2D(self.kernel_size, strides=self.strides, padding="same", name='pool')

#         self.B1_conv_pw = Conv2D_BN(filters = self.filters,
#                                     kernel_size=(1,1),
#                                     strides=1,
#                                     use_bias = False,
#                                     use_bn = True,
#                                     activation = self.act ,
#                                     name = 'pool')
        

#         self.B2_conv_pw = Conv2D_BN(filters = self.filters,
#                                     kernel_size=(1,1),
#                                     strides=1,
#                                     use_bias = False,
#                                     use_bn = True,
#                                     activation = self.act ,
#                                     name = 'conv_1')
        

#         self.B2_conv = Conv2D_BN(filters = self.filters,
#                                     kernel_size=(3,3),
#                                     strides=(2, 2),
#                                     use_bias = False,
#                                     use_bn = True,
#                                     activation = self.act,
#                                     name = 'conv_2')
            
#     def call(self, x):

#         x1 = self.B1_mp(x)
#         x1 = self.B1_conv_pw(x1)

#         x2 = self.B2_conv_pw(x)
#         x2 = self.B2_conv(x2)

#         out = Concatenate([x1, x2],axis=-1)
#         return out
    
    
#     def get_config(self):
#         config = super().get_config()
#         config.update(
#                 {
#                 "kernel_size": self.kernel_size,
#                 "strides": self.strides,
#                 "bn_epsilon" : self.bn_epsilon,
#                 "bn_momentum" : self.bn_momentum,
#                 "activation" : self.act
#                 }
#         )
#         return config
    
# #-------------------------------------------------------------------------------------------
# #
# #-------------------------------------------------------------------------------------------    
# class RepVGGBlock(tf.keras.layers.Layer):
#     VERSION = '1.0.0'
#     r""" Conv2D_Layer (conv2d_no_bias)
#     support fuse conv+bn as one conv layer


#     References:
#         - [Inspired by @hoangthang1607's  implementation of repvgg] (https://github.com/hoangthang1607/RepVGG-Tensorflow-2/blob/main/repvgg.py)
    
#     Args:
#         filters : int,,
#         kernel_size : int, defaults to 1
#         strides: int =1,
#         use_bias:bool= False,
#         groups: int =1,
#         use_bn : bool= True,
#         bn_epsilon : float= 1e-5,
#         bn_momentum : float= 0.9,
#         activation = None,
#         depoly : bool = False,
#         name : str='Conv2D'
#     Note:



#     Examples:

#     ```python
#     from tensorflow.keras.models import Model
#     from tensorflow.keras.layers import Input

#     x = Input(shape=(256,256,3))
#     out = Conv2D_BN(filters = 128, depoly = False, name = 'Conv2D_BN')(x)
#     model = Model(x, out)

#     x = Input(shape=(256,256,3))
#     out = Conv2D_BN(filters = 128, depoly = True, name = 'Conv2D_BN')(x)
#     deploy_model = Model(x, out)

#     """
#     def __init__(self,
#             filters : int,
#             kernel_size : int=1,
#             strides: int =1,
#             use_bias:bool= False,
#             groups: int =1,
#             use_bn : bool= True,
#             bn_epsilon : float= 1e-5,
#             bn_momentum : float= 0.9,
#             activation = 'relu',
#             deploy : bool = False,
#             name : str='Conv2D', **kwargs):
#         super().__init__(name=name)

        
#         if isinstance(kernel_size, (list, tuple)) :
#             self.kernel_size = kernel_size
#         else :
#             self.kernel_size = (kernel_size, kernel_size)

#         if self.kernel_size!=(3,3):
#             raise TypeError(f"now, only support kernel_size=3 @{self.__class__.__name__}"
#                            f"but got {self.kernel_size} ")

#         self.deploy = deploy
#         self.filters = filters
#         self.strides = strides
#         self.use_bias = True if self.deploy else use_bias
#         self.groups = groups
#         self.use_bn = use_bn
#         self.bn_epsilon = bn_epsilon
#         self.bn_momentum = bn_momentum

#         if not isinstance(activation,(str, type(None))):
#             raise TypeError("activation must be 'str' type like 'relu'"
#                         f"but got {type(activation)} ")
    
#         self.act = activation
        
#     def build(self, input_shape):
#         _,_,_, self.in_channels = input_shape

#         if not self.deploy:
#             if self.filters == self.in_channels and self.strides == 1:
#                 self.bn_identity = BatchNormalization(epsilon=self.bn_epsilon,
#                                                     momentum=self.bn_momentum,
#                                                     name="bn_identity")
        
#             self.conv_bn_1x1 =  Conv2D_BN(filters = self.filters,
#                                         kernel_size=1,
#                                         strides= self.strides,
#                                         use_bias = self.use_bias,
#                                         groups=self.groups,
#                                         use_bn = self.use_bn,
#                                         activation = None,
#                                         bn_epsilon = self.bn_epsilon,
#                                         bn_momentum = self.bn_momentum,
#                                         deploy = self.deploy,
#                                         name = 'Conv2D_bn_1x1')
            
#         self.conv_bn_3x3 =  Conv2D_BN(filters = self.filters,
#                                     kernel_size=self.kernel_size,
#                                     strides= self.strides,
#                                     use_bias = self.use_bias,
#                                     groups=self.groups,
#                                     use_bn = self.use_bn,
#                                     activation = None,
#                                     bn_epsilon = self.bn_epsilon,
#                                     bn_momentum = self.bn_momentum,
#                                     deploy = self.deploy,
#                                     name = 'Conv2D_bn_3x3')   
        
#         if self.act is not None  :
#             self.act = Activation(self.act, name=self.act)

#     def call(self,inputs):

#         feats = []
#         feats.append(self.conv_bn_3x3(inputs))
        
#         if hasattr(self,'bn_identity'):
#             id_out = self.bn_identity(inputs)
#             feats.append(id_out)

#         if hasattr(self,'conv_bn_1x1'):
#             feats.append(self.conv_bn_1x1(inputs))
        
#         out = tf.keras.layers.Add(name='Add')(feats)
#         if self.act is not None  :
#             out = self.act(out)
#         return out

#     def get_config(self):
#         config = super().get_config()
#         config.update(
#                 {
#                     "in_channels": self.in_channels,
#                     "filters": self.filters,
#                     "kernel_size": self.kernel_size,
#                     "strides": self.strides,
#                     "use_bias": self.use_bias,
#                     "groups": self.groups,
#                     "use_bn" : self.use_bn,
#                     "bn_epsilon" : self.bn_epsilon,
#                     "bn_momentum" : self.bn_momentum,
#                     "deploy" : self.deploy,
#                     "activation" : self.act.name if self.act is not None else  self.act
#                 }
#         )
#         return config
    
#     def _convert_bn_identity(self, bn_identity):

#         input_dim = self.in_channels // self.groups
#         kernel_value = np.zeros(
#                     (3, 3, input_dim, self.in_channels), dtype=np.float32
#         )
#         for i in range(self.in_channels):
#             kernel_value[1, 1, i % input_dim, i] = 1
        
#         self.id_tensor = tf.convert_to_tensor(
#                     kernel_value, dtype=np.float32
#         )
#         kernel = self.id_tensor
#         running_mean = bn_identity.moving_mean
#         running_var = bn_identity.moving_variance
#         gamma = bn_identity.gamma
#         beta = bn_identity.beta
#         eps = bn_identity.epsilon
        
#         std = tf.sqrt(running_var + eps)
#         t = gamma / std
#         return kernel * t, beta - running_mean * gamma / std
    
#     def _pad_1x1_to_3x3_tensor(self, kernel1x1):
#         if kernel1x1 is None:
#             return 0
#         else:
#             return tf.pad(
#                 kernel1x1, tf.constant([[1, 1], [1, 1], [0, 0], [0, 0]])
#             )
        
#     def weights_convert_to_deploy(self) ->Tuple[Tensor,Tensor]:

#         if self.deploy!=False:
#             raise ValueError("' to extract trained weights(trainning) to fuse RepConv2D block as single one conv"
#                              f" arg : 'depoly' must be False @{self.__class__.__name__}")
        
#         if hasattr(self,'bn_identity'):  
#             kernelid, biasid = self._convert_bn_identity(self.bn_identity)
#         else:
#             kernelid, biasid = 0., 0.

#         if hasattr(self.conv_bn_3x3,'weights_convert_to_deploy'):
#             #kernel3x3  , bias3x3 = self.conv_bn_3x3.weights_convert_to_deploy()['conv']
#             sub_layer_weights_dict  = self.conv_bn_3x3.weights_convert_to_deploy()
#             kernel3x3  , bias3x3 = sub_layer_weights_dict['conv']

#         if hasattr(self.conv_bn_1x1,'weights_convert_to_deploy'):
#             kernel1x1, bias1x1= self.conv_bn_1x1.weights_convert_to_deploy()['conv']

#         kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid
#         bias = bias3x3 + bias1x1 + biasid

#         'update new weights for conv_3x3_bn'
#         sub_layer_weights_dict['conv'] = [kernel, bias]

#         weights_map = dict()
#         weights_map['conv_bn_3x3'] = sub_layer_weights_dict
#         return  weights_map