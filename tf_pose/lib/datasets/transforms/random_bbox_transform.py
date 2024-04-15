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
class RandomBBoxTransform(CVBaseTransformLayer):
    version ='2.0.0'
    r"""Rnadomly shift and resize the bounding boxes.
    Author : Dr. David Shen
    Date : 2024/3/20

    Required Keys:
        - bbox
        - image_size
    Modified Keys:
        - bbox

    Args:
        shift_factor (float): Randomly shift the bbox in range
            :math:`[-dx, dx]` and :math:`[-dy, dy]` in X and Y directions,
            where :math:`dx(y) = x(y)_scale \cdot shift_factor` in pixels.
            Defaults to 0.16
        shift_prob (float): Probability of applying random shift. Defaults to
            0.3
        scale_factor (Tuple[float, float]): Randomly resize the bbox in range
            :math:`[scale_factor[0], scale_factor[1]]`. Defaults to (0.5, 1.5)
        scale_prob (float): Probability of applying random resizing. Defaults
            to 1.0
    """
    def __init__(self,
            shift_factor: float = 0.16,
            shift_prob: float = 0.3,
            scale_factor: Tuple[float, float] = (0.75, 1.25),
            scale_prob: float = 0.5, 
            test_mode : bool = False,
            **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.shift_factor = shift_factor
        self.shift_prob = shift_prob if not test_mode else 1.
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob if not test_mode else 1.

    
    def transform(self, data : Dict[str,Tensor],**kwargs) -> Dict[str,Tensor]:
        """The transform function of :class:`RandomBboxTransform`.
        See ``transform()`` method of :class:`BaseTransform` for details.
        Args:
            data (dict): The data dict
        Returns:
            dict: The result dict.
        """
        assert data.get('bbox') is not None, "Require Data Keys : 'bbox'  @RandomBBoxTransform"
        bbox = data["bbox"]

        cond_shift = tf.greater(self.shift_prob,tf.random.uniform(()))
        cond_scale = tf.greater(self.scale_prob,tf.random.uniform(()))

        # offset_wh = tf.random.uniform(shape=(2,), 
        #                 minval=-self.shift_factor,
        #                 maxval=self.shift_factor,
        #                 dtype=tf.dtypes.float32)
        
        offset_wh = tf.random.truncated_normal(
            shape=(2,),
            mean=0.0,
            stddev=0.5,
            dtype=self.compute_dtype
        )*self.shift_factor # between -1. and 1. with gaussain normal distribution

        offset_wh = tf.where(cond_shift, offset_wh, [0.,0.])

        scale_min, scale_max = self.scale_factor
        mu = (scale_min + scale_max) * 0.5
        sigma = (scale_max - scale_min) * 0.5

        # scale_wh = tf.random.uniform(
        #     shape=(), 
        #     minval=-1., 
        #     maxval=1.,
        #     dtype=tf.dtypes.float32)* sigma + mu
        
        scale_wh = tf.random.truncated_normal(
            shape=(),
            mean=0.0,
            stddev=0.5,
            dtype=self.compute_dtype
        )*sigma + mu

        scale_wh = tf.where(
            cond_scale, scale_wh, 1.
        )

        ' get bbox_center/bbox_scale to do trnasform'
        if data.get('bbox_center', None) is not None:
            bbox_ctr_xy = data['bbox_center']
        else:
            bbox_ctr_xy = bbox[:2] +  bbox[2:4]/2

        if data.get('bbox_scale', None) is not None:
            bbox_wh = data['bbox_scale']
        else:
            bbox_wh = bbox[2:4]

        bbox_ctr_xy +=  offset_wh*scale_wh
        bbox_wh =  bbox_wh*scale_wh

        # bbox_ctr_xy = bbox[:2] +  bbox[2:4]/2
        # bbox_ctr_xy +=   offset_wh*scale_wh
        # bbox_wh =  bbox[2:4]*scale_wh

        img_size_xy = tf.cast(
            tf.reverse(data['image_size'],axis=[0]) ,dtype=self.compute_dtype
        )
        bbox_xy_lt = tf.maximum(
            bbox_ctr_xy-bbox_wh/2, [0.,0.]
        )
        bbox_xy_rb = tf.minimum(
            bbox_ctr_xy+bbox_wh/2, img_size_xy-1.
        )
        bbox_xywh = tf.concat(
            [bbox_xy_lt, bbox_xy_rb-bbox_xy_lt],axis=-1
        )
        'update new bbox'
        data['bbox'] = bbox_xywh

        'update new bbox_center / bbox_scale if they already given'
        if data.get('bbox_center', None) is not None:
           data['bbox_center'] =  (bbox_xy_lt+bbox_xy_rb)/2.

        if data.get('bbox_scale', None) is not None:
           data['bbox_scale'] =  bbox_xywh[2:]


        return data 

