from collections.abc import Callable 
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import tensorflow as tf
from lib.models.backbones.base_backbone import BaseBackbone
from lib.Registers import MODELS
from .base_transformer import ClassToken, AddPositionEmbs, TransformerBlock
import typing_extensions as tx

ConfigDict = tx.TypedDict(
      "ConfigDict",
      {
          "dropout": float,
          "mlp_dim": int,
          "num_heads": int,
          "num_layers": int,
          "hidden_size": int,
      },
   )

############################################################################
#
#
############################################################################
@MODELS.register_module()
class ViT(BaseBackbone):

    CONFIG_B: ConfigDict = {
        "dropout": 0.1,
        "mlp_dim": 3072,
        "num_heads": 12,
        "num_layers": 12,
        "hidden_size": 768,
    }


    CONFIG_L: ConfigDict = {
        "dropout": 0.1,
        "mlp_dim": 4096,
        "num_heads": 16,
        "num_layers": 24,
        "hidden_size": 1024,
    }
    
    arch_settings = dict(vit_b = CONFIG_B, vit_l=CONFIG_L)

    def __init__(self,      
            model_input_shape : Tuple[int,int]=(256,192),
            patch_size: int=16,
            arch :  str = "vit_b", 
            class_token :bool = True,
            data_preprocessor: dict = None,
            pretrained : bool = False,
            pretrained_weights_path : str = None,
            pretrained_weights_skip_mismatch : bool = False,
            classes : int = 1000,  
            activation_top : str = "linear",
            include_top : bool = True,
            representation_size : Optional[int] = None, **kwargs):
    
        assert (model_input_shape[0] % patch_size == 0) and (
            model_input_shape[1] % patch_size == 0
        ), "image_size must be a multiple of patch_size"
    
        self.arch_setting = self.arch_settings[arch]
        self.dropout = self.arch_setting['dropout']
        self.mlp_dim = self.arch_setting['mlp_dim']
        self.num_heads = self.arch_setting['num_heads']
        self.num_layers = self.arch_setting['num_layers']
        self.hidden_size = self.arch_setting['hidden_size']
        self.use_class_token = class_token
        self.patch_size = patch_size

        'top head'
        self.classes = classes
        self.activation = activation_top
        self.include_top = include_top
        self.representation_size = representation_size

        super().__init__(input_size = (*model_input_shape,3) ,
                        data_preprocessor = data_preprocessor,
                        pretrained = pretrained,
                        pretrained_weights_path = pretrained_weights_path,
                        pretrained_weights_skip_mismatch  = pretrained_weights_skip_mismatch,
                        name = 'ViT')


    def forward(self, x:tf.Tensor)->tf.Tensor:
        

        x = tf.keras.layers.Conv2D(filters=self.hidden_size,
                    kernel_size=self.patch_size,
                    strides=self.patch_size,
                    padding="valid",name="embedding")(x) 

        x = tf.keras.layers.Reshape((x.shape[1] * x.shape[2], self.hidden_size))(x)
        x = ClassToken(name="class_token")(x)
        x = AddPositionEmbs(name="Transformer/posembed_input")(x) 

        if self.use_class_token == False:
            x = x[:,1:,:]

        for n in range(self.num_layers):
            x, _ = TransformerBlock(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                name=f"Transformer/encoderblock_{n}",
            )(x)

        x = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="Transformer/encoder_norm"
        )(x)

        y = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(x)

        if self.representation_size is not None:
            y = tf.keras.layers.Dense(self.representation_size, name="pre_logits", activation="tanh")(y)

        if self.include_top:
            y = tf.keras.layers.Dense(self.classes, name="head", activation=self.activation)(y)
        else:
            y = x

        return y

