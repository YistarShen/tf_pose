
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Sequence
import tensorflow as tf
import tensorflow_addons as tfa
from  .base import CVBaseTransformLayer
from lib.Registers import TRANSFORMS

@TRANSFORMS.register_module()

class RandomAffine(CVBaseTransformLayer):
    VESRION = '1.0.0'
    r""" RandomAffine

    Required Data's Keys:
        - img
        - img_shape
        - bbox (option)
        - kps (option)

    Modified Keys:
        - img
        - img_shape
        - bbox (option)
        - kps (option)

    References:
        - [Based on implementation of "RandomAffine" @mmdet] (https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/transforms/transforms.py)
        - [Inspired by "RandomPerspective" @ultralytics] (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py)
        - [Inspired by "get_rotation_matrix" @keras_cv] (https://github.com/keras-team/keras-cv/blob/master/keras_cv/utils/preprocessing.py)
        - [Inspired by "RandomShear" @keras_cv] (https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing/random_shear.py)

    Args:
        target_size (Tuple[int]): Output dimension after the transform, `[height, width]`.
                If `None`, output is the same size as input image . defaults to (640,640).
        fill_value (int) : a float represents the value to be filled outside the
                boundaries, defaults to 0
        interpolation (str) :  Interpolation mode. Supported values: "nearest","bilinear". Defaults to 'bilinear'
        max_rotate_degree(float) : Maximum degrees of rotation transform.  Defaults to 30.
        max_translate_ratio (float) :  gMaximum ratio of translation. Defaults to 0.1
        max_shear_degree (float) :  Maximum degrees of shear transform. Defaults to 2. 
        scaling_ratio_range (Tuple[float]) : Min and max ratio of scaling transform. Defaults to (0.75, 1.25). 
        rotate_along_with_iamge_center(bool) : whether to rotat image along with image's center point 
                if False, iamge will rotate along with (0,0) point, default to False
                if max_rotate_degree = 0, this arg will be invalid

    Note:
        - traslation matrix : 
        - scaling matrix  : 
        - shear matrix : 
        - ritation matrix  :

    
    Examples:
        rand_affine_cfg =  RandomAffine(
            target_size =(320,320),
            fill_value = 0,
            interpolation  = "bilinear",
            max_rotate_degree = 30.,
            max_translate_ratio = 0.1,
            max_shear_degree  = 10.,
            scaling_ratio_range = (0.75, 1.25),
            rotate_shear_along_with_img_center = True,
            align_to_img_center = True
        )

        
    """
    def __init__(
            self,
            target_size : Union[Tuple[int],int]=(640,640),
            fill_value : int = 0,
            interpolation : str = "bilinear",
            max_rotate_degree: float=30.,
            max_translate_ratio: float = 0.1,
            max_shear_degree : float = 10.,
            scaling_ratio_range: Tuple[float, float] = (0.75, 1.25),
            rotate_shear_along_with_img_center : bool = False,
            align_to_img_center : bool = True,
            **kwargs): 
        super().__init__(**kwargs)
    
        self.deg2rad = tf.cast(
            0.017453292519943295, dtype=self.compute_dtype
        )
        
        'args : tf.raw_ops.ImageProjectiveTransformV3 / tfa.image.transform'
        if isinstance(target_size, (list, tuple)) :
            self.target_size = target_size
        else :
            self.target_size = (target_size, target_size)

        self.interpolation = interpolation
        self.pad_val = fill_value
        'cfg '
        self.align_to_img_center = align_to_img_center
        self.scaling_ratio_range = scaling_ratio_range
        self.max_translate_ratio = max_translate_ratio
        self.max_shear_degree = max_shear_degree
        self.max_rotate_deg = max_rotate_degree
        if self.max_rotate_deg :
            self.rotate_shear_along_with_img_center = rotate_shear_along_with_img_center
        else:
            self.rotate_shear_along_with_img_center = False
        
        
    @staticmethod
    def _get_translation_matrix(
            trans_xy: float) -> tf.Tensor:
        r'''
        The transform matrix looks like:
            (1, 0, x)
            (0, 1, y)
            (0, 0, 1)
            where the last entry is implicit.

            We flatten the matrix to `[1., 0, x, 0., 1., y, 0., 0.]` for
            use with ImageProjectiveTransformV3.
        '''
        if trans_xy[0]==trans_xy[1]==0.:
            return tf.eye(3, dtype=trans_xy.dtype)
        # trans_xy : (b,3,1)
        translation_matrix = tf.cast(
            [
                [1,   0,   trans_xy[0]],
                [0,   1,   trans_xy[1]],
                [0,   0,   1          ]
            ],
            dtype=trans_xy.dtype
        )#[3,3]
        return translation_matrix

    @staticmethod
    def _get_rotation_matrix(
            rotate_rad: float, 
            img_wh : Optional[tf.Tensor] = None ) -> tf.Tensor:
        
        r'''
        The transform matrix looks like:
            (cos_p,   -sin_p,  x_offset)
            (sin_p,    cos_p,  y_offset)
            (    0,        0,         1)
            We flatten the matrix to `[cos_p, -sin_p, x_offset, sin_p, cos_p, y_offset, 0., 0.]` for
            use with ImageProjectiveTransformV3.
        '''
        """
        rotate_rad : (b,1,1)
        """
        compute_dtype = rotate_rad.dtype
        if rotate_rad==0.:
            return tf.eye(3, dtype=compute_dtype)
        
        cos_p = tf.cast( tf.math.cos(rotate_rad), dtype=compute_dtype) #(b,1,1)
        sin_p = tf.cast( tf.math.sin(rotate_rad), dtype=compute_dtype) #(b,1,1)
        if isinstance(img_wh, tf.Tensor):
            img_wh_tensor =  tf.cast(img_wh-1, dtype=compute_dtype)
            x_offset = (img_wh_tensor[0] - (img_wh_tensor[0]*cos_p - img_wh_tensor[1]*sin_p) )/2.
            y_offset = (img_wh_tensor[1] - (img_wh_tensor[0]*sin_p + img_wh_tensor[1]*cos_p) )/2.
        else:
            x_offset = y_offset = 0

  
        rotation_matrix = tf.cast(
            [
                [cos_p,  -sin_p,   x_offset],
                [sin_p,   cos_p,   y_offset],
                [0,           0,          1]
            ],
            dtype=compute_dtype
        )#[3,3]
        return rotation_matrix

    @staticmethod
    def _get_shear_matrix(
            xy_shear_rad: float,  
            img_wh : Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        
        r'''
        The transform matrix looks like:
            (1,       shear_x,  0)
            (shear_y,      1,   0)
            (0,            0,   1)
            We flatten the matrix to `[1.0, x, 0.0, y, 1.0, 0.0, 0.0, 0.0]` for
            use with ImageProjectiveTransformV3.
        '''
        """
        xy_shear_rad : (b,3,2)
        """
        compute_dtype = xy_shear_rad.dtype
        if xy_shear_rad[0] ==xy_shear_rad[1] == 0.:
            return tf.eye(3, dtype=compute_dtype)
        shear_xy = tf.cast(
            tf.math.tan(xy_shear_rad), dtype=compute_dtype
        )#(2,)

        if isinstance(img_wh, tf.Tensor):
            img_wh_tensor =  tf.cast(img_wh-1, dtype=compute_dtype)
            x_shift = -0.5*img_wh_tensor[1]*shear_xy[0] 
            y_shift = -0.5*img_wh_tensor[0]*shear_xy[1]
        else:
            x_shift = y_shift = 0


        shear_matrix = tf.cast(
            [
                [1,           shear_xy[0],  x_shift],
                [shear_xy[1],           1,  y_shift],
                [0,                     0,        1]
            ],
            dtype=compute_dtype
        )#[3,3]
        return shear_matrix
    

    @staticmethod
    def _get_scaling_matrix(
            scale_ratio: float) -> tf.Tensor:
        r'''
        The transform matrix looks like:
            (s, 0, 0)
            (0, s, 0)
            (0, 0, 1)
            where the last entry is implicit.

            We flatten the matrix to `[1.0, x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]` for
            use with ImageProjectiveTransformV3.
        '''
        compute_dtype = scale_ratio.dtype
        if scale_ratio == 1.:
            return tf.eye(3, dtype=compute_dtype)
    
        scaling_matrix = tf.cast(
            [
                [scale_ratio,         0,   0],
                [0,         scale_ratio,   0],
                [0,                   0,   1]
            ],
            dtype=compute_dtype
        )#[3,3]
        return scaling_matrix
    
    def _get_base_offset(
            self,  img_wh : tf.Tensor, target_size_wh : tf.Tensor
    )->tf.Tensor:
        if self.align_to_img_center:
            base_trans_xy = tf.cast(
                img_wh- target_size_wh,
                dtype=self.compute_dtype
            )*0.5       
        else:
            base_trans_xy = tf.zeros_like(img_wh, dtype=self.compute_dtype)
        return base_trans_xy

    def _get_random_homography_matrix(self, img_hw : tf.Tensor):
        # get basic cfg
        target_szie_wh = tf.cast(
             [self.target_size[1],self.target_size[0]],dtype=tf.int32
        ) 
        img_wh = tf.reverse(img_hw, axis=[0])
        base_trans_xy = self._get_base_offset(img_wh, target_szie_wh)

        'init tf_matAffine_3x3'
        tf_matAffine_3x3 =  tf.eye(3, dtype=self.compute_dtype)

        '#1 Translation'
        trans_xy = self._rand_val(
            shape=(2,), 
            minval=-self.max_translate_ratio, 
            maxval=self.max_translate_ratio
        )
        # trans_xy = tf.random.uniform(
        #             shape=(2,),
        #             minval = -self.max_translate_ratio,
        #             maxval = self.max_translate_ratio,
        #             dtype = self.compute_dtype
        # )

        #trans_xy *= tf.cast(img_wh, dtype=self.compute_dtype)
        trans_xy = base_trans_xy + trans_xy*tf.cast(img_wh, dtype=self.compute_dtype)
        translate_matrix  = self._get_translation_matrix(
                trans_xy
        )
        tf_matAffine_3x3 = tf.linalg.matmul(
            tf_matAffine_3x3,
            translate_matrix,
        )
        '#2 Rotation'
        rotation_degree =  self._rand_truncated_normal(
                shape=(), mean=0.0, stddev=0.5,
        )*self.max_rotate_deg
        # rotation_degree = tf.random.uniform(
        #         shape=(),
        #         minval = -self.max_rotate_deg,
        #         maxval = self.max_rotate_deg,
        #         dtype = self.compute_dtype
        # )
        rotation_matrix = self._get_rotation_matrix(
                rotation_degree*self.deg2rad ,
                img_wh =  target_szie_wh if self.rotate_shear_along_with_img_center else None
        )
        tf_matAffine_3x3 = tf.linalg.matmul(
                    tf_matAffine_3x3,
                    rotation_matrix,
        )
        '#3 Scaling'
        scaling_ratio = self._rand_val(
            shape=(), 
            minval=self.scaling_ratio_range[0], 
            maxval=self.scaling_ratio_range[1]
        )
        # scaling_ratio = tf.random.uniform(
        #             shape=(),
        #             minval=self.scaling_ratio_range[0],
        #             maxval=self.scaling_ratio_range[1],
        #             dtype = self.compute_dtype
        # )
  
        scaling_matrix = self._get_scaling_matrix(
                scaling_ratio
        )
        tf_matAffine_3x3 = tf.linalg.matmul(
                    tf_matAffine_3x3,
                    scaling_matrix,
        )
        '#4 Shear'

        xy_degree = self._rand_truncated_normal(
            shape=(2,), mean=0.0, stddev=0.5
        )*self.max_shear_degree

        # xy_degree = tf.random.uniform(
        #         shape=(2,),
        #         minval = -self.max_shear_degree,
        #         maxval= self.max_shear_degree,
        #         dtype = self.compute_dtype
        # )
        shear_matrix = self._get_shear_matrix(
                xy_degree*self.deg2rad,
                img_wh =  target_szie_wh if self.rotate_shear_along_with_img_center else None
        )
        tf_matAffine_3x3 = tf.linalg.matmul(
                    tf_matAffine_3x3,
                    shear_matrix,
        )
        return tf_matAffine_3x3
    

    def tf_xy_point_affine_transform(self, pt_xy, matAffine_2x3):
        """ 
        pt_xy           pt_xy                   associated
        shape_rank      shape                   matAffine_2x3   
        --------------------------------------- ---------------------------------------------------
        3            (b1, b2, 2)               matAffine_2x3[None,None,...]     
        2            (b1,2,)                   matAffine_2x3[None,...]      
        1            (2,)                      matAffine_2x3  
        """
        pt_shape_rank = pt_xy.shape.rank #(1,2,3)
        for _ in range (pt_shape_rank-1):
            matAffine_2x3 = matAffine_2x3[None,...]

        pt_3x1 = tf.expand_dims( 
            tf.concat(
                [pt_xy[...,:2], tf.ones_like(pt_xy[...,:1])],
                axis=-1
            ) ,
            axis=-1
        ) #(points,3,1) 
        new_pt = tf.linalg.matmul(matAffine_2x3,pt_3x1) #(17,2,1)  (None,2,3)@(points,3,1)=  (None,3,1)
        return tf.squeeze(new_pt, axis=-1)
    
    def get_bboxes(
            self, 
            data, 
            matAffine_2x3, 
            target_size_yx=(640,640)
        ):   
        bbox_xywh = data["bbox"]  #(num_bbox, 4), (4,)
        bbox_lt = bbox_xywh[...,:2]
        bbox_rb = bbox_lt[...,:2] + bbox_xywh[...,2:]
        bbox_rt = tf.stack(
            [bbox_rb[...,0], bbox_lt[...,1]], axis=-1
        )
        bbox_lb = tf.stack(
            [bbox_lt[...,0], bbox_rb[...,1]], axis=-1
        ) #(num_bbox,2)
        pts = tf.stack(
            [bbox_lt,bbox_rb,bbox_rt,bbox_lb], axis=0
        ) #(4, num_bbox, 2)
        #print(pts.shape.rank, matAffine_2x3.shape.rank)
        new_pts = self.tf_xy_point_affine_transform(
            pts,matAffine_2x3
        )#(num_bbox,4,2)
        bboxes_xy_rb = tf.reduce_max(
                new_pts, axis=0
        )  #(num_bbox,2)  
        bboxes_xy_lt = tf.reduce_min(
                new_pts, axis=0
        )  #(num_bbox,2)  
        bboxes_xy_lt = tf.math.maximum(
            x = bboxes_xy_lt,
            y = 0.
        )
        bboxes_xy_rb = tf.math.minimum(
            x = bboxes_xy_rb,
            y = (target_size_yx[1],target_size_yx[0])
        )
        bboxes_wh = bboxes_xy_rb - bboxes_xy_lt 
        mask = tf.math.reduce_all(
            tf.greater(bboxes_wh,0.), axis=-1
        ) #(b,None) , true mean valid, false is invalid
        # print(max_pts_xy.shape, min_pts_xy.shape)
        bboxes_xywh= tf.concat(
            [bboxes_xy_lt,bboxes_wh], axis=-1
        )
        data['bbox'] = tf.boolean_mask(
            bboxes_xywh, mask
        ) 
        if data.get('labels',None)is not None :
            data['labels'] = tf.boolean_mask(
                data['labels'], mask
            )
        if data.get('gt_mask',None)is not None :
            data['gt_mask'] = mask
        return data 
    
    def get_kps(self,
                data, 
                matAffine_2x3, 
                target_size_yx=(640,640)):
        
        kps = data["kps"] #(num_kps,3) / (none, num_kps,3)
        vis = kps[...,2:3] #(num_kps,1) / (none, num_kps,1)
      
        new_pts = self.tf_xy_point_affine_transform(
            kps[...,:2],matAffine_2x3
        ) #(17,2)

        'filter out ksp outside image after affine transform'
        mask_max = tf.math.less_equal(
            new_pts[...,:2], [target_size_yx[1], target_size_yx[0]]
        )
        mask_min = tf.math.greater_equal( 
            new_pts[...,:2], [0., 0.]
        )
        mask_vis = tf.greater(vis, 0) if vis.shape!=0 else []
        mask = tf.concat([mask_max, mask_min, mask_vis], axis=-1)  #(17,5)
        mask = tf.math.reduce_all(mask, axis=-1, keepdims=True) #(17,1)
        new_pts = tf.where(mask, new_pts[...,:2], tf.cast(0.,dtype=self.compute_dtype) )       #(17,2)
        vis = tf.cast(mask, dtype=self.compute_dtype)        #(17,1)
        data["kps"]  = tf.concat([new_pts, vis],axis=-1)
        return data

    def transform(
            self,data : Dict[str,tf.Tensor],**kwargs) -> Dict[str,tf.Tensor]:

        #image = data["image"]
        'formatting image type'
        image = self.img_to_tensor(data["image"]) 
        #-----------------------------------------------------------------------------------------------
        src_img_shape = tf.shape(image)[:2]
        warp_matrix = self._get_random_homography_matrix(src_img_shape)
        flatten_matrix_8x1 = tf.reshape(warp_matrix, shape=(9,))[:8]
        if False :
            tf_dst_img = tfa.image.transform(
                image,
                tf.cast(flatten_matrix_8x1, dtype=tf.float32),
                interpolation=self.interpolation.lower(),
                fill_mode='constant',
                output_shape = src_img_shape if self.target_size==(None,None) else self.target_size 
            )#image_size_yx
        else:
            tf_dst_img = tf.raw_ops.ImageProjectiveTransformV3(
                images = image[None,...],
                transforms = tf.cast(flatten_matrix_8x1[None,...],dtype=tf.float32),
                fill_value=tf.convert_to_tensor(0., tf.float32),
                fill_mode ="CONSTANT",
                interpolation = self.interpolation.upper(), 
                output_shape =  src_img_shape if self.target_size==(None,None) else self.target_size                            
            )
            tf_dst_img = tf_dst_img[0, :, : ,:]
        #-----------------------------------------------------------------------------------------------
        'update img and modify its type to data'
        data = self.update_data_img(tf_dst_img, data)  #tf.tensor->tf.ragged_tensor or tf.tensor->tf.tensor

        #tf_trans_inv = tf.linalg.inv(warp_matrix) # (3,3)
        tf_trans_inv = tf.linalg.inv(
            tf.cast(
                warp_matrix, 
                dtype= self.compute_dtype if tf.__version__ > '2.10.0' else tf.float32
            )
        ) # (3,3)
        matAffine_2x3 = tf.cast( 
            tf_trans_inv[:2,:] , dtype=self.compute_dtype
        )#(2,3)

        if data.get('bbox', None) is not None :
            data = self.get_bboxes(
                data, matAffine_2x3, self.target_size
            )
        if data.get('kps', None) is not None :
            data = self.get_kps(
                data ,matAffine_2x3, self.target_size
            )

        return data