
import tensorflow as tf
from tensorflow import Tensor
from typing import Dict, List, Optional, Tuple, Union
from lib.Registers import TRANSFORMS
from .base import CVBaseTransformLayer, VectorizedTransformLayer

############################################################################
#
#
############################################################################
# @TRANSFORMS.register_module()
# class EnsureTensor():
#     version = '1.0.0'
#     r"""Ensure training Data Type is tf.Tensor 
#     if values' data type is tf.RaggedTensor, modify them to tf.Tensor before encoding data 
    

#     Modified Data's Keys :
#         - all in data

#     """
#     def __call__(
#             self,
#             data : Dict[str,Tensor]
#         ) -> Dict[str,Tensor]:

#         if not isinstance(data, dict):
#             raise TypeError(
#                 "data_type must be dict "
#                 f"but got type {type(data)} @{self.__class__.__name__}"
#             )

#         data = self.dict_to_tensor(data)
#         return data
        
#     def dict_to_tensor(self, 
#                 data : Dict[str,Tensor]) -> Dict[str,Tensor]:
    
#         for key, val in data.items():
#             if key not in ['meta_info', 'transform2src']:
#                 if not isinstance(val, (tf.Tensor)):
#                     data[key] = val.to_tensor() 
#         return data
    
@TRANSFORMS.register_module()
class EnsureTensor(VectorizedTransformLayer):
    version = '1.0.0'
    r"""Ensure training Data Type is tf.Tensor 
    if values' data type is tf.RaggedTensor, modify them to tf.Tensor before encoding data 
    

    Modified Data's Keys :
        - all in data

    """
    def __init__(self, *arg,**kwargs):
        super().__init__(*arg, **kwargs)

    def batched_transform(self,data: Dict,  *args, **kwargs):
        for key, val in data.items():
            if key not in ['meta_info', 'transform2src']:
                if not isinstance(val, (tf.Tensor)):
                    data[key] = val.to_tensor() 
        return data
        