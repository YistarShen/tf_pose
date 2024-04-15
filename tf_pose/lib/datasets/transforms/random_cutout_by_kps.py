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
class RandomCutoutByKeypoints(CVBaseTransformLayer):
    version = '2.0.1'
    'RandomCutoutByKeypoints'
    r"""Rnadomly Keypoints Dropout
    Date : 2024/3/24
    Author : Dr. Shen 

    Bug : float16 cannot work !!!!!!!!!!!!!!!!!!!!!!!

    it should be applied after cropping/resize images


    Required Keys:
        - kps
        - image
        - bbox
    Modified Keys:
        - image

    Args:
        base_mask_ratio (float): the ratio of  mask_area/bbox_area
            - Defaults to 0.05 
        mask_scale_factor (Tuple[float, float]): Randomly Scale for applied mask area.
            - Defaults to (0.8, 1.2) 
        min_bbox_area (float) : if bbox area is less than it, return data without ksp dropout.
            - Defaults to 64*64                   
        prob (float): Probability of applying random mask.
            Defaults to 0.3
        drop_joints_indices :Tuple[int] = [5,6,7,8,11,12],

    """
    def __init__(self, 
            base_mask_ratio : float = 0.05,
            mask_scale_factor : Tuple[float, float] = (0.8, 1.2), 
            min_bbox_area : float = 64.*64.,
            prob : float = 0.3, 
            drop_joints_indices :Tuple[int] = [5,6,7,8,11,12],
            random_mask_color : bool = False, 
            test_mode : bool =False,              
            **kwargs): 
        super().__init__(**kwargs)

        self.ratio = base_mask_ratio
        self.prob = prob if not test_mode else 1.
        self.drop_joints_indices = drop_joints_indices
        self.mask_scale_factor = mask_scale_factor
        self.random_mask_color = random_mask_color
        self.min_bbox_area = min_bbox_area

    def gather_eff_kps(self,kps_17x3):
        candiate_kps = tf.gather(kps_17x3, self.drop_joints_indices, axis=0)
        cond = tf.greater(candiate_kps[:,2], 0. )
        return tf.boolean_mask(candiate_kps,cond)


    def transform(self, data : Dict[str,Tensor], **kwargs) -> Dict[str,Tensor]:
        """The transform function of :class:`RandomBboxTransform`.
        See ``transform()`` method of :class:`BaseTransform` for details.
        Args:
            data (dict): The data dict
        Returns:
            dict: The result dict.
        """
        # assert data.get('image') is not None, "Require Data Key : 'image'  @RandomKPSDropout"
        # assert data.get('kps') is not None, "Require Data Key : 'kps'  @RandomKPSDropout"
        # assert data.get('bbox') is not None, "Require Data Key : 'bbox'  @RandomKPSDropout"

        self.verify_required_keys(data,['image', 'kps', 'bbox'])

        if self.prob < tf.random.uniform(()):
            return data
        
        'formatting image type'
        image = self.img_to_tensor(data["image"]) 
        #image = data["image"]
        bbox = data["bbox"]
        kps_17x3 = data["kps"]

        bbox_area = tf.reduce_prod(bbox[2:])
        if bbox_area < self.min_bbox_area :
            return data
        
        
        image_size_yx = tf.shape(image)[:2]
        #image_size_yx = data['image_size']

        eff_kps = self.gather_eff_kps(kps_17x3)
        eff_kps_num = tf.shape(eff_kps)[0]
    
        if eff_kps_num==0:
            return data
        
        idx = tf.random.uniform(shape=(),minval=0,maxval=eff_kps_num,dtype=tf.int32)
        kp_int = tf.cast(eff_kps[idx,:3],dtype=tf.int32) 

        'random select id'
        random_scale = tf.random.uniform(shape=(),
                        minval = self.mask_scale_factor[0],
                        maxval = self.mask_scale_factor[1],
                        dtype = self.compute_dtype)
    
        #half_mask_size = tf.math.sqrt(tf.reduce_prod( tf.cast(image_size_yx, dtype=tf.float32) )*self.ratio)/2

        half_mask_size = tf.math.sqrt(bbox_area*self.ratio)/2
        half_mask_size = random_scale*half_mask_size

        random_offset = tf.random.uniform(shape=(2,), 
                        minval=-0.25,
                        maxval=0.25,
                        dtype=self.compute_dtype
        )*half_mask_size     
        random_offset = tf.cast(random_offset, dtype=tf.int32)

        half_mask_size = tf.cast(half_mask_size, dtype=tf.int32)
        full_mask_size = 2*half_mask_size+1
        mask = tf.zeros(shape=(full_mask_size,full_mask_size), dtype=tf.uint8)
    
        lu = tf.reverse( tf.cast( kp_int[:2]+random_offset - half_mask_size, dtype=tf.int32), axis=[0]) 
        rb = tf.reverse( tf.cast( kp_int[:2]+random_offset + half_mask_size+1, dtype=tf.int32), axis=[0]) 


        if (lu[0]>=image_size_yx[0] or lu[1]>=image_size_yx[1] or rb[0]<0 or rb[1]<0 or kp_int[2]==0):
            boolean_map_HxW = tf.ones(shape=image_size_yx,dtype=tf.uint8)
            #boolean_map_HxWx3 = tf.ones(shape=image_size_yx,dtype=tf.uint8)
        else:
            x_left = tf.math.maximum(0, -lu[1])
            x_right = tf.math.minimum(rb[1], image_size_yx[1]) - lu[1]
            y_up = tf.math.maximum(0, -lu[0])
            y_bottom = tf.math.minimum(rb[0], image_size_yx[0]) - lu[0]  
            eff_mask = mask[y_up:y_bottom, x_left:x_right]
            paddings = tf.stack([ lu, image_size_yx-rb],  axis=-1)
            paddings = tf.math.maximum([[0, 0], [0, 0]], paddings)
            boolean_map_HxW = tf.pad(eff_mask, paddings, constant_values=1)

        boolean_map_HxWx3 = tf.tile(boolean_map_HxW[:,:,None],[1,1,3])
        image = image*boolean_map_HxWx3
    
        if self.random_mask_color :
            color = tf.cast( tf.random.uniform(shape=(), minval=0, maxval=255,dtype=tf.int32), dtype=tf.uint8)
            cond = tf.equal(boolean_map_HxWx3,0)
            image = tf.where(cond,color,image)

        'update transformed image'    
        #data['image'] = aug_img
        data = self.update_data_img(image, data)
    
        return data
    