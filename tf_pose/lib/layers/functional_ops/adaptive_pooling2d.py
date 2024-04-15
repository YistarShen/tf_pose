'tf layers'
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization,LayerNormalization, Activation, DepthwiseConv2D, Dense, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import ZeroPadding1D, ZeroPadding2D, Concatenate, Multiply, Reshape
from tensorflow import Tensor
import tensorflow as tf


class AdaptiveAveragePooling2D(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r"""
    
    """
    def __init__(self, output_size,  name=None, **kwargs):
        super(AdaptiveAveragePooling2D, self).__init__(name = name)
        #super(layer_split, self).__init__(**kwargs)
        self.output_size = output_size

    def build(self, input_shapes):

        #n, h, w, c = x.get_shape().as_list()
        self.n, self.h, self.w, self.c = input_shapes.as_list()

    def call(self, x):
        #x:(b,h,w,c)
        h_bins = self.output_size[0]
        w_bins = self.output_size[1]

        split_cols_list = tf.split(x, h_bins, axis=1)  #tensor list [(b, h/h_bins, w, c), .... ] 
        split_cols = tf.stack(split_cols_list, axis=1) # (b, h_bins, h/h_bins, w, c)
        split_rows_list = tf.split(split_cols, w_bins, axis=3) #tensor list [(b, h/h_bins, w/w_bins, c), .... ] 
        split_rows = tf.stack(split_rows_list, axis=3)   # (b, h_bins, h/h_bins, w_bins, w/w_bins, c)
        #out_vect = self.reduce_function(split_rows, axis=[2, 4])
        out_vect = 0.
        _, _, h_bins, _, w_bins, _= split_rows.shape
        for i in range (h_bins):
            for j in range (w_bins):
                out_vect += split_rows[:,:,i,:,j,:]
        return out_vect/(h_bins+w_bins)
    
    def get_config(self):
        config = super(AdaptiveAveragePooling2D, self).get_config()
        config.update({"output_size": self.output_size,})
        return config  