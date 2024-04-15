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
class RandomGaussianBlur(CVBaseTransformLayer):
    version = '1.0.0'
    r"""Randomly Gaussian Blur for Image.

    Ref : https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing/random_gaussian_blur.py
    
    Required Data's Keys:
        - img
    
    Args:
        kernel_size_xy(Tuple[int]): 2 element tuple or 2 element list. x and y dimensions
                for the kernel used. If tuple or list, first element is used for the
                x dimension and second element is used for y dimension. If int,
                kernel will be squared.  
                Defaults=(5,5)
        factor (float) : Mathematically, `factor` represents the `sigma`
                value in a gaussian blur. `factor=0.0` makes this layer perform a
                no-op operation, and high values make the blur stronger.  
                Defaults to 3.
        prob (float): The  probability of Gaussian Blur op. 
                Defaults to 0.5

    TO DO: support batch image

    """
    def __init__(self, 
                kernel_size_xy : Tuple[int]=(3,3),
                factor : float =0.5,
                prob : Optional[float]=0.5,
                test_mode :bool=False,
                **kwargs): 
        super().__init__(**kwargs)
        
        if isinstance(kernel_size_xy, (tuple, list)):
            self.x = kernel_size_xy[0]
            self.y = kernel_size_xy[1]
        else:
            if isinstance(kernel_size_xy, int):
                self.x = self.y = kernel_size_xy
            else:    
                raise ValueError(
                    "`kernel_size` must be list, tuple or integer "
                    ", got {} ".format(type(kernel_size_xy))
                )
        self.factor = factor
        self.prob = prob
        self.test_mode = test_mode

    def get_kernel(factor, filter_size):
        x = tf.cast(
                    tf.range(-filter_size // 2 + 1, filter_size // 2 + 1),
                    dtype=tf.float32
        )  
        blur_filter = tf.exp(
            -tf.pow(x, 2.0)
            / (2.0 * tf.pow(tf.cast(factor, dtype=tf.float32), 2.0))
        ) 
        blur_filter /= tf.reduce_sum(blur_filter)
        return blur_filter

    def transform(
            self, data : Dict[str,Tensor], **kwargs
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
        
        'formatting image type'
        image = self.img_to_tensor( 
            tf.expand_dims(data["image"], axis=0), 
            dtype=tf.float32
        ) 
        # image = self.img_to_tensor(data["image"]) 
        # image = tf.expand_dims(data["image"], axis=0)
        # image = tf.cast(image, dtype=tf.float32)

        num_channels = tf.shape(image)[-1]
        if not self.test_mode:
            sigma = tf.random.uniform(())*self.factor
            sigma = tf.math.maximum(sigma, tf.keras.backend.epsilon())
        else:
            sigma = self.factor 

        blur_v = RandomGaussianBlur.get_kernel(sigma, self.y)
        blur_v = tf.reshape(blur_v, [self.y, 1, 1, 1])
        blur_h = RandomGaussianBlur.get_kernel(sigma, self.x)
        blur_h = tf.reshape(blur_h, [1, self.x, 1, 1])

        blur_h = tf.cast(
            tf.tile(blur_h, [1, 1, num_channels, 1]), dtype=tf.float32
        )
        blur_v = tf.cast(
            tf.tile(blur_v, [1, 1, num_channels, 1]), dtype=tf.float32
        )
        image = tf.nn.depthwise_conv2d(
            image, blur_h, strides=[1, 1, 1, 1], padding="SAME"
        )
        image = tf.nn.depthwise_conv2d(
            image, blur_v, strides=[1, 1, 1, 1], padding="SAME"
        )
        
        'update transformed image'    
        image = tf.squeeze(image, axis=0)
        data = self.update_data_img(image, data)

        return data
