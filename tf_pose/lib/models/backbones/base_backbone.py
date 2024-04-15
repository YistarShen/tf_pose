import os 
import tensorflow as tf
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from tensorflow import Tensor
from tensorflow.keras.layers import  Input
from lib.Registers import LAYERS
from lib.utils.common import is_path_available




from abc import ABC, abstractmethod
class BaseBackbone(tf.keras.Model):
    VERSION = '2.0.0'
    R"""BaseBackBone
    This class defines the basic functions of a backbone. Any backbone that
    inherits this class should at least define its own `forward` function.

    tf.keras.Model : supported_kwargs = [
                        "inputs",
                        "outputs",
                        "name",
                        "trainable",
                        "skip_init",
                    ]

    Args:
        input_size (Tuple[int,int]) : 
        data_preprocessor (dict) : default to None
        pretrained_weights_path (str) : default to None
        load_weights_skip_mismatch (bool) : default to False
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in DepthwiseConvModule and ConvModule, defaults to 'silu'.
        depoly (bool): determine depolyment config for each cell . default to None, 
                depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfgs depend on above args.
                depoly = True => to use deployment config, only conv layer will be bulit
                depoly = False => to use training config()
        
    References:
            - [Based on implementation of 'tf.keras.Model'] 
            (https://github.com/keras-team/keras/blob/v2.15.0/keras/engine/training.py#L542)
            - [Inspired by Backbone @keras_cv] 
            (https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/backbones/backbone.py#L25)     

    Note :
       - 
       - 
       - 
       - 
    name : str ='tf_model'

    """
    def __init__(
        self,
        input_size : Tuple[int,int],
        data_preprocessor: dict = None,
        pretrained_weights_path : Optional[str] = None,
        load_weights_skip_mismatch : bool = False,
        bn_epsilon : float= 1e-5,
        bn_momentum : float= 0.9,
        activation : str = 'relu',
        deploy  : Optional[bool] = None,
        *args, **kwargs
    ) -> tf.keras.Model:

        self.deploy = deploy
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.act_name = activation  


        self.pretrained_weights = pretrained_weights_path
        self.data_preprocessor = data_preprocessor

        model_input = Input(shape=input_size)
        preprocessor_layer = self.parse_preprocessor(self.data_preprocessor)
        model_output = self.call( 
            preprocessor_layer(model_input) if preprocessor_layer is not None else model_input
        ) 

        ''' 
        the kwargs for inheritance
        supported_kwargs = [
            "inputs",
            "outputs",
            "name",
            "trainable",
            "skip_init",
        ]
        '''
        super().__init__(
            inputs = model_input,
            outputs = model_output ,
            *args,  **kwargs
        )
        
    
        self.pretrained_weights_path = pretrained_weights_path
        self.load_pretrained_weights(
            self.pretrained_weights_path, 
            load_weights_skip_mismatch
        )   

 
    def load_pretrained_weights(
            self, 
            pretrained_weights_path : str = None, 
            skip_mismatch : bool = int
        ):

        if not pretrained_weights_path:  #pretrained_weights_path is None
            return
        
        if isinstance(pretrained_weights_path, str): 
            is_path_available(pretrained_weights_path)
            super().load_weights(
                filepath = pretrained_weights_path,
                by_name = True,
                skip_mismatch = skip_mismatch
            ) 
            print(
                f'already load weights from {pretrained_weights_path} with skip_mismatch={skip_mismatch}'
            )  
        else:
            raise TypeError(
                "backbone's pretrained_weights_path must be 'str' type"
                f"but got '{type(pretrained_weights_path)}'"
            )
            
             
    def parse_preprocessor(self, 
                        data_preprocessor : dict = None):
        'TO List to compose multilayers, i.e. reszie layer-> img_norm_layer'
        if data_preprocessor is not None:
            if isinstance(data_preprocessor, dict):
                preprocessor_layer = LAYERS.build(data_preprocessor)
            elif isinstance(data_preprocessor, (tf.Module, tf.keras.layers.Layer)):
                'directly use module or layer'
                preprocessor_layer = data_preprocessor
            else:
                raise TypeError(
                    "type of data_preprocessor must be 'dict', 'tf.Module' or 'tf.keras.layers.Layer'"
                    f"defualt to None   @{self.__class_.__name__}"
                )
        else:
            preprocessor_layer = None
        return preprocessor_layer



    @property
    def with_pretrained_weights(self) -> bool:
        """bool: whether the pose estimator has a head."""
        return hasattr(self, 'pretrained_weights_path') and os.path.exists(self.pretrained_weights_path) 
    
    @classmethod
    def from_config(cls, config):
        # The default `from_config()` for functional models will return a
        # vanilla `keras.Model`. We override it to get a subclass instance back.
        return cls(**config)   
         
    def Output_TensorSpecs(self)->List:
        return self.model.outputs
    
    def Input_TensorSpecs(self)->List:
        return self.model.inputs
    
    @abstractmethod
    def forward(self, x : Tensor,
                is_train: bool=True):
        """Forward function.
        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """
        return NotImplemented
