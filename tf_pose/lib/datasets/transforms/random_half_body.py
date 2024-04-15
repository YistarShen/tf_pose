import tensorflow as tf
from tensorflow import Tensor
from typing import Dict, List, Optional, Tuple, Union
from lib.Registers import TRANSFORMS
from  .base import CVBaseTransformLayer


############################################################################
#
#
############################################################################
#https://blog.csdn.net/qq_43312130/article/details/122034420
@TRANSFORMS.register_module()
class RandomHalfBody(CVBaseTransformLayer):
    version = '2.1.0'
    r"""Data augmentation with half-body transform that keeps only the upper or lower body at random.

    due to apply tf.boolean_mask ,  using CVBaseTransformLayer (not VectorizedTransformLayer) to implement
    is more easy and clear , current this transform only support kps format with coco pose style (17kps)
 
    Date : 2024/3/25
    Author : Dr. David Shen 
    
    Required Data's Keys:
        - image_size
        - bbox 
        - kps

    Modified Keys:
        - bbox

    Added Keys :
        - bbox_center 
        - bbox_scale

    Args:
        min_total_keypoints (int): The minimum required number of total valid
            keypoints of a person to apply half-body transform. Defaults to 8
        min_half_keypoints (int): The minimum required number of valid
            half-body keypoints of a person to apply half-body transform.
            Defaults to 2
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.5
        prob (float): The probability to apply half-body transform when the
            keypoint number meets the requirement. Defaults to 0.3

    TO DO: support batch image
    """
    def __init__(self,
                prob : float = 0.3, 
                padding : float = 1.2,
                min_total_keypoints :  int = 9, 
                min_upper_keypoints :  int = 4,
                min_lower_keypoints :  int = 4,
                upper_prioritized_prob : float = 0.7,
                test_mode : bool = False,             
                **kwargs): 
        super().__init__(**kwargs)

        'base cfg'
        self.prob = 1. if test_mode else prob
        self.upper_prioritized_prob = upper_prioritized_prob

        self.padding = padding
        'num_ksp_thr'
        self.min_upper_keypoints = min_upper_keypoints
        self.min_lower_keypoints = min_lower_keypoints
        self.min_total_keypoints = min_total_keypoints

        self.upper_index = tf.constant([0,1,2,3,4,5,6,7,8],dtype=tf.int32)
        self.lower_index = tf.constant([9,10,11,12,13,14,15,16],dtype=tf.int32)

    def transform(self,  data : Dict[str,Tensor],**kwargs) -> Dict[str,Tensor]:

        assert 'kps' and 'image_size' and 'bbox' in data.keys(), \
        "Required Data Keys :  'kps', 'image_size' and 'bbox' "
        #assert data['image_size'].shape.rank == 3, 'not yet support batch data support'
        #assert data['bbox'].shape.rank == 1, 'not yet support multi bboxes'
        
        bbox = data["bbox"]
        if( bbox[2]*bbox[3]<64.*64. or self.prob<tf.random.uniform(()) ):
            return data
        
        'only consider coco 17kps to determine new bbox'
        vis =  tf.not_equal(data['kps'][:,2], 0.) 
        upper_kps_num = tf.reduce_sum(tf.cast(vis[:9],dtype=tf.int32))
        lower_kps_num = tf.reduce_sum(tf.cast(vis[9:17],dtype=tf.int32))
        
        if (upper_kps_num+lower_kps_num)<self.min_total_keypoints:
            return data 

        'upper. and  lower kps both are less than 5'
        if(upper_kps_num<=self.min_upper_keypoints and lower_kps_num<=self.min_lower_keypoints):
            return data
      
        if tf.random.uniform(()) < self.upper_prioritized_prob: #.  select upper ksp
            if upper_kps_num>self.min_upper_keypoints :
                half_body_index = tf.boolean_mask(self.upper_index, vis[:9]) 
            else:
                half_body_index = tf.boolean_mask(self.lower_index, vis[9:17]) 
        else:
            if lower_kps_num>self.min_lower_keypoints :
                half_body_index = tf.boolean_mask(self.lower_index, vis[9:17]) 
            else: 
                half_body_index = tf.boolean_mask(self.upper_index, vis[:9]) 

        half_body_kps = tf.gather(data["kps"][:17, :], half_body_index, axis=0)[...,:2]
        
        'calculate new bbox'
        center = tf.reduce_mean( half_body_kps, axis=0)
        left_top = tf.math.reduce_min(half_body_kps, axis=0)
        right_bottom = tf.math.reduce_max(half_body_kps, axis=0)
    
        'to fit image shape; left-top point > (0.,0.) and wh < bbox[2:]'
        wh = (right_bottom - left_top)*self.padding

        img_size_xy = tf.cast( 
            tf.reverse(data['image_size'],axis=[0]), dtype=self.compute_dtype
        )
        thr_wh = tf.math.minimum(img_size_xy-center-1., center-1.)
        wh = tf.math.minimum(thr_wh*2., wh)

        'filter out too small transformed bbox( no change bbox)'
        if(tf.reduce_prod(wh)<96*96):
            return data
        
        xy = tf.math.maximum( self.fp_zero, center-wh/2)
        half_body_bbox = tf.concat([xy,wh],axis=-1) #new bbox_xywh

        data["bbox"] = half_body_bbox

        'add new keys (to do) optinal'
        #data["bbox_center"] = center
        #data["bbox_scale"] = wh
        return data
