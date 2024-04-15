
import tensorflow as tf
from tensorflow import Tensor
from typing import Dict, List, Optional, Tuple, Union
from lib.Registers import TRANSFORMS
from  .base import CVBaseTransformLayer, VectorizedTransformLayer

############################################################################
#
#
############################################################################
@TRANSFORMS.register_module()
class VectorizedIdendity(VectorizedTransformLayer):
    version = '1.0.'
    r"""Idendity Transform Test

    Required Keys:
        - img
        - bbox 
        - kps (optional)

    Modified Keys:
        - img
        - bbox_scale

    Added Keys:
        - input_size
        - transformed_keypoints
     
    Args: 
        prob (float) :  xxx probability.
        test_mode (bool):
    """
    def __init__(self,
                prob = 0.5,
                test_mode=False,
                **kwargs): 
        super().__init__(**kwargs)
        'base cfg'
        self.test_mode = test_mode
        'prob and params'
        self.prob = prob

    def batched_transform(self, data : Dict[str,Tensor], *args, **kwargs) -> Dict[str,Tensor]:
        "test code"
        if self.test_mode :
           image = data['image']
           image += 1

        return data 
    
############################################################################
#
#
############################################################################
@TRANSFORMS.register_module()
class Idendity(CVBaseTransformLayer):
    version = '1.0.'
    r"""Idendity Transform Test

    Required Keys:
        - img
        - bbox 
        - kps (optional)

    Modified Keys:
        - img
        - bbox_scale

    Added Keys:
        - input_size
        - transformed_keypoints
     
    Args: 
        prob (float) :  xxx probability.
        test_mode (bool):
    """
    def __init__(self,
                prob = 0.5,
                test_mode=False,
                **kwargs): 
        super().__init__(**kwargs)
        'base cfg'
        self.test_mode = test_mode
        'prob and params'
        self.prob = prob
 
    def add_Datakeys(self,data : Dict[str,Tensor]) -> Dict[str,Tensor]:
        return data
    
    def transform(self, data : Dict[str,Tensor],*args, **kwargs) -> Dict[str,Tensor]:

        "test code"
        if self.test_mode :
            image = data['image']
            image += 1
            if not isinstance(image, tf.Tensor):
                image =  image.to_tensor()

            if isinstance(data['image'], tf.RaggedTensor):
                data['image'] =  tf.RaggedTensor.from_tensor(image)  

        return data    
    


# ############################################################################
# #
# #
# ############################################################################
# @TRANSFORMS.register_module()
# class IdendityTest(CVBaseTransformLayer):
#     version = '1.0.'
#     r"""Idendity Transform Test

#     Required Keys:
#         - img
#         - bbox 
#         - kps (optional)

#     Modified Keys:
#         - img
#         - bbox_scale

#     Added Keys:
#         - input_size
#         - transformed_keypoints
     
#     Args: 
#         prob (float) :  xxx probability.
#         test_mode (bool):
#     """
#     def __init__(self,
#                 prob = 0.5,
#                 test_mode=False,
#                 **kwargs): 
#         super().__init__(**kwargs)
#         'base cfg'
#         self.test_mode = test_mode
#         'prob and params'
#         self.prob = prob
 
#     def add_Datakeys(self,data : Dict[str,Tensor]) -> Dict[str,Tensor]:
#         return data
    
#     def transform(self, data : Dict[str,Tensor],**kwargs) -> Dict[str,Tensor]:
#         image = data['image']
#         if self.test_mode :
#            image += 1
#         if not isinstance(image, tf.Tensor):
#            image =  image.to_tensor()
        
#         if isinstance(data['image'], tf.RaggedTensor):
#            data['image'] =  tf.RaggedTensor.from_tensor(image)  

#         return data 
    