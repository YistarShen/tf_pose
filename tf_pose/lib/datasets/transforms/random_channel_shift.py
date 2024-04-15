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
class RandomChannelShift(CVBaseTransformLayer):
    version = '2.0.0'
    r"""Randomly shift values for each channel of the input image(s).
    Author : Dr. David Shen
    Date : 2024/3/26

    Ref : https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing/random_channel_shift.py
    
    Required Data's Keys:
        - img
    
    Args:
        value_range (Tuple[int]): 2 element tuple or 2 element list. x and y dimensions
                for the kernel used. If tuple or list, first element is used for the
                x dimension and second element is used for y dimension. If int,
                kernel will be squared.  
                Defaults=(0,255)
        factor (float) : A scalar value in the range `[0.0, 1.0]`. If `factor` is a single value, it will
                interpret as equivalent to the tuple `(0.0, factor)`. The `factor`
                will sample between its range for every image to augment. 
                Defaults to 0.5
        channels (int) : the number of channels to shift, defaults to 3 which
                corresponds to an RGB shift. In some cases, there may ber more or
                less channels.
                Defaults to 3
        prob (float): The  probability of Gaussian Blur op. 
                Defaults to 0.5

    TO DO: support batch image

    """
    def __init__(
            self, 
            value_range : Tuple[int]=(0,255), 
            factor : Union[Tuple[float], List[float],float] = (0.3, 0.8),
            prob : Optional[float]=0.5,
            channels : int = 3,
            test_mode :bool=False,
            **kwargs): 
        super().__init__(**kwargs)
        
        if isinstance(factor,(Tuple,list)):
            self.factor = factor
        else:
            if isinstance(factor, float):
                if factor<0.:
                    raise   ValueError( "`factor` must be >0."
                    f", got {factor} ")
    
                self.factor = (0., factor if factor<1. else 1.)
            else:
                raise   TypeError(
                    "`factor` must be list, tuple  or float "
                    ", got {} ".format(type(factor))
                )
    
        #self.factor = tf.cast(factor, dtype=tf.float32)
        self.value_range = value_range
        self.prob = prob
        self.channels = channels
        self.test_mode = test_mode
    
    def get_required_keys(self):
        return {'image'}
        

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
        
        image_fp = self.img_to_tensor(
            data["image"], dtype=self.compute_dtype
        ) 

        shift_rgb =self._rand_val( 
            shape = [self.channels],  
            minval = self.factor[0],
            maxval = self.factor[1],
        )
        # shift_rgb = tf.random.uniform(
        #     shape = [self.channels], 
        #     minval = self.factor[0], 
        #     maxval = self.factor[1], 
        #     dtype = self.compute_dtype
        # )
        shift_rgb = shift_rgb*self._rand_inverse(shape=())*0.5 #(3,)

        image_fp = image_fp + shift_rgb[None,None,:]*255. #(3,)
        #tf.print(image_fp.dtype)
        image_fp = tf.clip_by_value(image_fp, self.value_range[0], self.value_range[1])
         
        'update image'
        data = self.update_data_img(image_fp, data)
        return data
