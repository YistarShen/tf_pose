
import tensorflow as tf
from tensorflow import Tensor
from typing import Dict, List, Optional, Tuple, Union
from lib.Registers import TRANSFORMS
#from  .base import CVBaseTransform

@TRANSFORMS.register_module()
class ImageNormalize():
    version = '1.0.0'
    r"""Image Normalize / rescale.
    support batched image transform
    
    Required Keys:
        - img

    Required Data dtype:
        - tf.Tensor 

    Args:
        prob (float): The flipping probability. Defaults to 0.5

  
    """
    def __init__(self,
          img_mean = [0.485, 0.456, 0.406],  
          img_std = [0.229, 0.224, 0.225]):
        
        self.img_mean = tf.constant(img_mean, dtype=tf.float32)
        self.img_std = tf.constant(img_std, dtype=tf.float32) 

    def __call__(self,data : Dict, inverse=False) ->Dict:
        """ it can fully support batched transform""" 

        if not isinstance(data, dict):
            raise TypeError(
                "data_type must be dict "
                f"but got type {type(data)} @{self.__class__.__name__}"
            )
        
        
        if (data['image'].shape.rank == 3 or data['image'].shape.rank == 4) and data['image'].shape[-1]==3:
            image = tf.cast(data['image'], dtype=tf.float32)/255.
            data['image'] = (image - self.img_mean)/self.img_std
        else:
            raise TypeError(
                "image'shape must be (b, None,None,3) or (None,None,3)"
                f"but got type {data['image'].shape} @{self.__class__.__name__}"
            )         
        return data
