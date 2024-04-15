# Copyright (c) Movella All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Sequence
from tensorflow import Tensor
import tensorflow as tf


#---------------------------------------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------------------------------------
class BaseCodec(tf.keras.layers.Layer):
    VERSION = '2.2.0'
    R"""Base class for all BaseCodec that inherit 'tf.keras.layers.Layer'
    Author : Dr. David Shen
    Date : 2024/3/28

    encode type : ensure numerical stability in encode type, we force dtype to tf.float32


    Note:
        - decode type : it will decode inputs to model's input image coordinate frame
        - encode type : ensure numerical stability
        - encode type : data[''y_true'], data['sample_weight'], data['y_pred'] must be keep as dtype=float32 for stability in training progress 

    
    """
    def __init__(
        self, 
        use_vectorized_map : bool = False,
        parallel_iterations : int = 16,
        codec_type : str = 'encode',  
        dtype = tf.float32,
        **kwargs 
    ):

        super().__init__(dtype = dtype, **kwargs)
        self.use_vectorized_map = use_vectorized_map
        self.parallel_iterations = parallel_iterations
        self.embedded_codec = False

        self.codec_type = codec_type.lower()
        if self.codec_type not in ['encode','decode']:
            raise ValueError(
                f"codec_type must be 'encode' or 'decode' @{self.__class__.__name__}, "
                f"but got codec_type : {self.codec_type}"
            ) 
     

        
       

    def count_params(self):
        # The label encoder has no weights, so we short-circuit the weight
        # counting to avoid having to `build` this layer unnecessarily.
        return 0   
    
    def op_assert_all_finite(
            self, tensor : tf.Tensor, message : str = 'Input x must be all finite'
    ):  
        return tf.debugging.assert_all_finite(
            tensor, message = message + f" __({tensor.dtype})"
        )
      
    def op_expand_batch_dim(
            self, data : Union[Tensor, Sequence[Tensor], Dict[str,Tensor]]
    ):
        return tf.nest.map_structure(
            lambda x : tf.expand_dims(x, axis=0), data
        )
    
    def op_copy_dict_data(self, data : dict):
        if not isinstance(data, dict):
            raise TypeError(
                "op_copy_dict_data only support 'dict' type, "
                f", but got {type(data)}"
            )    
        return {k:v for k,v in data.items()}

    def op_cast(
            self, data : Union[Tensor, Sequence[Tensor], Dict[str,Tensor]],
            dtype = tf.float32
        ):
        return  tf.nest.map_structure(
            lambda x : tf.cast(x, dtype=dtype), data
        )
    
    def op_ensure_fp32(
            self, data : Union[Tensor, Sequence[Tensor], Dict[str,Tensor]]
        ):
        return  tf.nest.map_structure(
            lambda x : tf.cast(x, dtype=tf.float32) if x.dtype!=tf.float32 else x, data
        )
    
    def op_assert_dtype(
            self, data : Union[Tensor, Sequence[Tensor], Dict[str,Tensor]], message='tensor', dtype= tf.float32
        ):
        return  tf.nest.map_structure(
            lambda x : tf.debugging.assert_type(x, dtype, message=message), data, 
        )  
    

    def batch_encode(self, data: dict, y_pred : Optional[Tensor]=None) -> dict:
        r"""Encode data

        Note:
            -
           
        Args:
            data (dict[str,Tensor]): 
   
        Returns:
           Encoded items (dict[str,Tensor]) : .
        """
        raise NotImplementedError()
    

    def batch_decode(self, data: Tensor, *args, **kwargs) -> Any:
        r"""Decode data.

        Note:
            - Now only support type of data is 'Tensor'
            -
            

        Args:
            data (Union[Tensor,Dict]): 


        Returns:
            Decoded items (Union[Tensor,Dict]) :
        """
        raise NotImplementedError()
    
    
    def call(
        self, 
        inputs : Union[Tensor, Sequence[Tensor], Dict[str,Tensor]],  
        codec_type : str = 'encode',  
        *args, **kwargs
    ):   
        
        r""" 
        
        
        """
        if codec_type.lower() == 'encode':
            data = self.batch_encode(
                inputs, 
                y_pred=kwargs['y_pred'] if 'y_pred' in kwargs else None,
            )
            # ensure y_true, y_pred, sample_weight is dtype=tf.float32 in training progress
            # 1.) y_true
            if data.get('y_true',None) is None :
                raise TypeError(
                    "when codec_type = 'encode', output dict data of codec layer must contain 'y_true' key"
                    f", but got {data.keys()} @ {self.__class__.__name__}"
                )   
        
            data['y_true'] = self.op_ensure_fp32(data['y_true'])
            # 2.) sample_weight (option)
            if data.get('sample_weight',None) is not None :
                data['sample_weight'] = self.op_ensure_fp32(data['sample_weight']) 
            # 3.) sample_weight (option)
            if data.get('y_pred',None) is not None :
                data['y_pred'] = self.op_ensure_fp32(data['y_pred'])

        elif codec_type.lower() == 'decode': 
            'support input dtype : tensor, tuple[tensor], list[Tensor] and Dict[str,Tensor]'
            if not isinstance(inputs,dict):
                data = dict(y_pred=inputs)
            else:
                if inputs.get('y_pred', None) is None :
                    raise ValueError(
                        f"if codec_type= 'decode'and input type is dict, must be 'encode' or 'decode' @{self.__class__.__name__}, "
                        f"the key 'y_pred' is neccssary , bit got keys : {inputs.keys()}"
                    ) 
              
            data = self.batch_decode(
                data, *args, **kwargs
            )

            if data.get('decode_pred',None) is None :
                raise TypeError(
                    "when codec_type = 'decode', output dict data of codec layer must contain 'decode_pred' key"
                    f", but got {data.keys()} @ {self.__class__.__name__}"
                )   

            if not isinstance(inputs,dict): 
                data  = data['decode_pred']
            
        else:
            raise ValueError(
                f"codec_type must be 'encode' or 'decode' @{self.__class__.__name__}, "
                f"but got codec_type : {codec_type}"
            ) 
        
        return data

    def gen_tfds_w_codec(
            cls, 
            tfrec_datasets_list : Union[List, Dict], 
            transforms_list  : Union[List, Dict], 
            test_mode=True, 
            batch_size=16, 
            shuffle=True
    ):
        from lib.datasets import dataloader
        if not isinstance(tfrec_datasets_list, list):
            tfrec_datasets_list = [tfrec_datasets_list]
        if not isinstance(transforms_list, list):
            transforms_list = [transforms_list]     

        
        tfds_builder = dataloader(
            batch_size = batch_size,
            prefetch_size = batch_size,
            shuffle  =  shuffle,
            tfrec_datasets_list = tfrec_datasets_list,
            augmenters = transforms_list,
            codec = cls,
            parallel_iterations = batch_size
        )
        tfds_builder.get_pipeline_cfg(note=' @codec self test') 
        return  tfds_builder.GenerateTargets(    
            test_mode=test_mode, 
            unpack_x_y_sample_weight= False, 
            ds_weights =None
        )  

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedded_codec": self.embedded_codec,

            }
        )
        return config  
                 