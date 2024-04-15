
from collections.abc import Callable 
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from lib.Registers import MODELS
import tensorflow as tf
from tensorflow.keras import layers

@MODELS.register_module()
class PoseLifterHead(tf.keras.layers.Layer):
    def __init__(self,
            num_joints : int= 17, 
            name=None, 
            **kwargs): 
        
        self.num_joints = num_joints
        self.bias_init = tf.random_normal_initializer(stddev=0.06)
        'Head '
        self.out_dims = self.num_joints*3
       
    def __call__(self, inputs):

        x = layers.LayerNormalization(epsilon=1e-6, name="Head_norm", dtype='float32')(inputs)  #(b,f,17*dims)
        reg_xyz = layers.Dense(self.out_dims, bias_initializer=self.bias_init, name="Head/Dens_Reg_Output", dtype='float32')(x) #(b,f,17*3)
        #conf = layers.Dense(self.num_joints, bias_initializer=self.bias_init, name="Head/Dens_Conf_Output", dtype='float32')(x) #(b,f,17)
        reg_xyz = layers.Reshape((-1,self.num_joints,3), name="Head/reg_Reshape", dtype='float32')(reg_xyz) #(b,f,17*3) => (b,f,17,3)

        return reg_xyz