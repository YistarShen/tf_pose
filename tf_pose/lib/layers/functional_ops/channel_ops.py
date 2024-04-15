
'tf layers'
# from typing import Any, Callable, Dict, List, Optional, Tuple, Union
# from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization,LayerNormalization, Activation, DepthwiseConv2D, Dense, MaxPooling2D, GlobalAveragePooling2D
# from tensorflow.keras.layers import ZeroPadding1D, ZeroPadding2D, Concatenate, Multiply, Reshape
# from tensorflow import Tensor
# import tensorflow as tf


# class ShuffleChannel(tf.keras.layers.Layer):
#     VERSION = '1.0.0'
#     r"""
    
#     """
#     def __init__(self, groups, name=None,**kwargs):
#         super(ShuffleChannel, self).__init__(name = name)
#         self.groups = groups
#         #super(shuffle_Channel, self).__init__(**kwargs)
        
#     def get_config(self):
#         config = super(ShuffleChannel, self).get_config()
#         config.update({"groups": self.groups})
#         return config  
    
#     def build(self, input_shapes):
#         #n, h, w, c = x.get_shape().as_list()

#         self.n, self.h, self.w, self.c = input_shapes.as_list()

#     def call(self, x):

#         x = tf.reshape(x, shape=[tf.shape(x)[0], self.h, self.w, self.groups, self.c // self.groups])  
#         x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))  
#         output = tf.reshape(x, shape=[tf.shape(x)[0], self.h, self.w, self.c])

#         return output

# #------------------------------------------------------------------------
# #
# #--------------------------------------------------------------------------
# class layer_split(tf.keras.layers.Layer):
#     VERSION = '1.0.0'
#     r"""
    
#     """
#     def __init__(self, 
#                 num_or_size_splits, 
#                 axis, 
#                 name=None, **kwargs):
#         super(layer_split, self).__init__(name = name)
#         self.num_or_size_splits = num_or_size_splits
#         self.axis = axis

#     def build(self, input_shapes):

#         #n, h, w, c = x.get_shape().as_list()
#         self.n, self.h, self.w, self.c = input_shapes.as_list()

#     def call(self, x):
#         output = tf.split(x, self.num_or_size_splits, axis=self.axis)
#         return output
    
#     def get_config(self):
#         config = super(layer_split, self).get_config()
#         config.update({"num_or_size_splits": self.num_or_size_splits,
#                          "axis": self.axis       
#                     })
#         return config           