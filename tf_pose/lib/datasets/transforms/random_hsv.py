import tensorflow as tf
from tensorflow import Tensor
from typing import Dict, List, Optional, Tuple, Union
from lib.Registers import TRANSFORMS
from  .base import CVBaseTransformLayer

############################################################################
#
#
############################################################################
@TRANSFORMS.register_module()
class RandomHSVAug(CVBaseTransformLayer):
    version = '1.0.0'
    r"""Rnadomly shift and resize the bounding boxes.
    it should be applied after cropping/resize images

    Required Keys:
        - image
    Modified Keys:
        - image

    Args:
        base_mask_ratio (float): Randomly shift the bbox in range
        mask_scale_factor (Tuple[float, float]): Randomly shift the bbox in range
        prob (float): Probability of applying random mask. Defaults to 0.3
        scale_factor (Tuple[float, float]): Randomly resize the bbox in range
            :math:`[scale_factor[0], scale_factor[1]]`. Defaults to (0.5, 1.5)
        scale_prob (float): Probability of applying random resizing. Defaults
            to 1.0
    """
    def __init__(self, 
            hue_prob :float = 0.5,
            saturation_prob :float = 0.5,
            brightness_prob : float = 0.5,
            hue_delta : float = 0.3, 
            saturation_factor: Tuple[float] = (0.5,2.0),
            brightness_delta  : float = 0.4,             
            **kwargs): 
        super().__init__(**kwargs)
        
        assert saturation_factor[0] < saturation_factor[1], \
        'saturation_factor_upper must be larger than saturation_factor_lower'
        assert hue_delta<=0.5 and hue_delta >0., \
        'hue_delta must be in [0., 0.5] '
        assert brightness_delta <= 1., \
        'brightness_delta must be <1.0 '

        """
        hue_delta = 0.3
        saturation_factor_lower=0.5,
        saturation_factor_upper=2.0,
        brightness_delta = 0.2
        """
        self.hue_prob = hue_prob
        self.saturation_prob = saturation_prob
        self.brightness_prob = brightness_prob

        self.hue_delta = hue_delta
        self.saturation_lower = saturation_factor[0] 
        self.saturation_upper = saturation_factor[1] 
        self.brightness_delta = brightness_delta  


    def transform(self, data : Dict[str,Tensor],**kwargs) -> Dict[str,Tensor]:

        'formatting image type'
        image = self.img_to_tensor(data["image"]) 
        #image = data["image"]
        if self.hue_delta > tf.random.uniform(()):
            image = tf.image.random_hue(
                image, 
                max_delta=self.hue_delta
            )  
        if self.saturation_prob > tf.random.uniform(()):
            image = tf.image.random_saturation(
                image, 
                lower=self.saturation_lower, 
                upper=self.saturation_upper
            )
        if self.brightness_prob > tf.random.uniform(()):
            image = tf.image.random_brightness(
                image, 
                max_delta=self.brightness_delta
            )
        'update transformed image'    
        data = self.update_data_img(image, data)
        #data["image"] = image
        return data
