
from collections.abc import Callable 
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers
from lib.models.backbones.base_backbone import BaseBackbone
from lib.Registers import MODELS
from ..layers.base_transformer import ClassToken, AddPositionEmbs, TransformerBlock



@MODELS.register_module()
class PoseFormerV1(BaseBackbone):
    VERSION = '1.0.0'
    r""" PoseFormerV1
    
    """
    def __init__(self, 
                num_frame :int = 27, 
                num_joints : int = 17, 
                in_chans :int = 2,
                embed_dim_ratio : int = 32, 
                depth : int=4,
                num_heads : int=8, 
                mlp_ratio : int=2., 
                qkv_bias=True,
                qk_scale=None,
                drop_rate : float=0., 
                attn_drop_rate: float=0., 
                drop_path_rate: float=0.):
    
        self.bias_init=tf.random_normal_initializer(stddev=0.06)

        self.num_frame = num_frame
        self.in_chans = in_chans
        self.num_joints = num_joints

    
        'transformer block' 
        self.embed_dim_ratio = embed_dim_ratio
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.num_heads =num_heads
        'drop_rate'
        self.pos_drop_rate = drop_rate
        self.mlp_drop_rate = drop_rate 
        self.attn_drop_rate = attn_drop_rate 
        self.drop_path_rate = tf.linspace(start=0., stop=drop_path_rate, num=depth)
        self.drop_path_rate = tf.cast(self.drop_path_rate, dtype=tf.float16).numpy()
        'Head '
        self.out_dims = self.num_joints*3


        model_input_shape = (self.num_frame, self.num_joints, self.in_chans)
        super().__init__(input_size = model_input_shape,
                        data_preprocessor = None,
                        pretrained = False,
                        pretrained_weights_path = None,
                        pretrained_weights_skip_mismatch  = True,
                        name = 'PoseFormerV1')
    
    def Block(self, encoded_patches, drop_path_rate, name="block"):

        embed_dim = encoded_patches.shape[-1] 

        x1 = layers.LayerNormalization(epsilon=1e-6, name=name+f"/MHSA/ln")(encoded_patches)
        '----------MHSA--------------------------------------------------------------------'
        attention_output = layers.MultiHeadAttention(
                        num_heads = self.num_heads, 
                        key_dim = (embed_dim//self.num_heads), 
                        use_bias = True,
                        bias_initializer = self.bias_init,
                        dropout=self.attn_drop_rate,
                        name = name+"/MHSA/attn_out",
                        )(x1, x1)
        'drop out'
        attention_output = layers.Dropout(drop_path_rate, name=name+"/MHSA/path_drop")(attention_output)
        'skip add'
        x2 = layers.Add(name=name+"/MHSA/skip_add")([attention_output, encoded_patches])

        '---------------MLP block----------------------------------------------------------'
        x3 = layers.LayerNormalization(epsilon=1e-6, name=name+f"/Mlp/ln")(x2)

        x3 = layers.Dense(embed_dim*self.mlp_ratio,bias_initializer=self.bias_init,activation=tf.nn.gelu, name=name+"/Mlp/hidden_dense")(x3)
        x3 = layers.Dropout(self.mlp_drop_rate, name=name+"/Mlp/hidden_feat_drop")(x3)
        x3 = layers.Dense(embed_dim, bias_initializer=self.bias_init, name=name+"/Mlp/dense")(x3)
        x3 = layers.Dropout(self.mlp_drop_rate, name=name+"/Mlp/out_feat_drop")(x3)
        'drop out'
        x3 = layers.Dropout(drop_path_rate, name=name+"/Mlp/path_drop")(x3)
        out = layers.Add(name=name+"/Mlp/skip_add")([x3, x2])
        return out

    def Spatial_forward_features(self, x):
        '''
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )
        '''
        # x:(b,f,17,2)

        x = tf.reshape(x,(-1,self.num_joints, self.in_chans)) #(b*f,17,2)

        x = layers.Dense(self.embed_dim_ratio,  bias_initializer=self.bias_init, name="Spatial_patch_to_embedding/Dens")(x) #(b*f,17,32)

        x = AddPositionEmbs(name="Spatial_pos_embed")(x)

        x = layers.Dropout(self.pos_drop_rate, name="Spatial_patch_to_embedding/pos_drop")(x) #(b*f,17,32)

        for i in range(self.depth):
            x = self.Block(x, self.drop_path_rate[i], name=f"spatial_block-{i+1}") #(b*f,17,32)
        
        
        x = layers.LayerNormalization(epsilon=1e-6, name="spatial_norm")(x)  #(b*f,17,32)
                                        
        x = tf.reshape(x,(-1,self.num_frame, self.num_joints*self.embed_dim_ratio)) #(b,f,17*32)


        return x

    def Temporal_forward_features(self, x):
        x = AddPositionEmbs(name="Temporal_pos_embed/AddPositionEmbs")(x)

        x = layers.Dropout(self.pos_drop_rate, name="Temporal_pos_embed/pos_drop")(x) #(b,f, 17*dims)

        for i in range(self.depth):
            x = self.Block(x, self.drop_path_rate[i], name=f"Temporal_block-{i+1}") #(b,f,17*dims)

        x = layers.LayerNormalization(epsilon=1e-6, name="Temporal_norm")(x)  #(b,f,17*dims)

        x = tf.keras.layers.Permute((2,1),name="Temporal_permute-1")(x)  #(b,17*dims,f)

        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = tf.keras.layers.Conv1D(1,kernel_size=1,strides=1,name="Temporal_weighted_mean/Conv1D")(x) #(b,17*dims,1)

        x = tf.keras.layers.Permute((2,1),name="Temporal_permute-2")(x)  #(b,1,17*dims)

        return x
    
    def forward(self, x:tf.Tensor)->tf.Tensor:
      
        x = self.Spatial_forward_features(x)
    
        x = self.Temporal_forward_features(x)

        return  x

    def Head(self,x):
        x = layers.LayerNormalization(epsilon=1e-6, name="Head_norm", dtype='float32')(x)  #(b,1,17*dims)
        reg = layers.Dense(self.out_dims, bias_initializer=self.bias_init, name="Head/Dens_Reg_Output", dtype='float32')(x) #(b,1,17*3)
        #conf = layers.Dense(self.num_joints, bias_initializer=self.bias_init, name="Head/Dens_Conf_Output", dtype='float32')(x) #(b,1,17)
        reg = layers.Reshape((1,self.num_joints,-1), name="Head/reg_Reshape", dtype='float32')(reg) #(b,1,17*3) => (b,1,17,3)

        return reg
      
