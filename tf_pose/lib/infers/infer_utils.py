import tensorflow as tf 
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


def resize_imgae_with_Pad(image : tf.Tensor, 
                        target_shape_yx : Tuple[int] =(640,640)):
    
    r"""resize_imgae_with_Pad by keeping aspect ratio 
    this function resizes imgae by keeping the aspect ratio,

    more detail can be seen as below
    tf.image.resize_with_pad(
        image,
        target_height,
        target_width,
        method=ResizeMethod.BILINEAR,
        antialias=False
    )
    Resizes an image to a target width and height by keeping the aspect ratio the same without distortion. 
    If the target dimensions don't match the image dimensions, the image is resized and then padded with zeroes to match requested dimensions

    """
    return tf.image.resize_with_pad(image, target_height=target_shape_yx[0],target_width=target_shape_yx[1])

