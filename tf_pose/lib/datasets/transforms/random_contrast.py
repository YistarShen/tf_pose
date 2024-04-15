import tensorflow as tf
from tensorflow import Tensor
from typing import Dict, List, Optional, Tuple, Union
from lib.Registers import TRANSFORMS
from  .base import CVBaseTransformLayer

   
############################################################################
#
#BaseTransform
############################################################################
@TRANSFORMS.register_module()
class RandomContrast(CVBaseTransformLayer):
    version = '1.0.0'
    r"""Randomly adjusts image's contrast.

    Ref : https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing/random_contrast.py
    
    Required Data's Keys:
        - img
    
    Args:
        value_range (Tuple[int]): A tuple or a list of two elements. The first value
                represents the lower bound for values in passed images, the second
                represents the upper bound. Images passed to the layer should have
                values within `value_range`.
                Defaults=(0,255)
        factor (float) : A positive float represented as fraction of value, or a tuple of
                size 2 representing lower and upper bound. When represented as a
                single float, lower = upper. The contrast factor will be randomly
                picked between `[1.0 - lower, 1.0 + upper]`. For any pixel x in the
                channel, the output will be `(x - mean) * factor + mean` where
                `mean` is the mean value of the channel.
                Defaults to 3.
        prob (float): The  probability to adjusts image's contrast. 
                Defaults to 0.5

    TO DO: support batch image

    """
    def __init__(
            self, 
            value_range : Tuple[int]=(0,255), 
            factor : Tuple[float]=(0.3,0.3),
            prob : Optional[float]=0.5,
            test_mode :bool=False,              
            **kwargs
    ): 
        super().__init__(**kwargs)

        if isinstance(factor, (tuple, list)):
            if factor[0]>factor[1]:
                raise ValueError(
                    f"` `factor[0]` must be <= `factor[0][1]`"
                    f"  Got `factor={factor}`"
                )

            self.factor_min = 1 - factor[0]
            self.factor_max = 1 + factor[1]
        else:
            self.factor_min = 1 - factor
            self.factor_max = 1 + factor

        if self.factor_min < -1: self.factor_min=-1
        if self.factor_max > 2: self.factor_max=2

        self.value_range = value_range
        self.prob = prob if not test_mode else 1.
        self.test_mode = test_mode


    def transform(
            self, data : Dict[str,Tensor],**kwargs
    ) -> Dict[str,Tensor]:
        """The transform function of :class:`RandomFlip`.
        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            data (dict):  The result dict
            i.e. data = {"image": img_tensor, "bbox":bbox_tensor, "kps": kps_tensor}
        Returns:
            dict: The output result dict. like input data
        """
        #assert data["image"].shape.rank==3, 'input data is batch type, not yet support batched transform'
        if  self.prob < tf.random.uniform(()):
            return data  
        
       
        'method-2'
        'formatting image type'
        images = self.img_to_tensor(data["image"].to_tensor(),dtype=tf.float32) 
        means = tf.reduce_mean(images, axis=[0, 1], keepdims=True)

        contrast_factors = tf.random.uniform(
            shape=(1,1,1),
            minval=self.factor_min,
            maxval=self.factor_max,
            dtype=tf.float32
        )
        if self.test_mode:
            contrast_factors = tf.ones(shape=(1,1,1), dtype=tf.float32)*self.factor_max
        images = (images - means)*contrast_factors + means
        images = tf.clip_by_value(
            images, self.value_range[0], self.value_range[1]
        )
        # 'update image'
        data = self.update_data_img(images, data)
        return data
