
import tensorflow as tf
from tensorflow import Tensor
import tensorflow_addons as tfa
from typing import Dict, List, Optional, Tuple, Union
from lib.Registers import TRANSFORMS
from  .base import CVBaseTransformLayer
from lib.codecs.keypoints import tf_clean_kps_outside_bbox, tf_kps_clip_by_img
from lib.codecs.bounding_box import fix_bboxes_aspect_ratio

@TRANSFORMS.register_module()
class TopdownAffine(CVBaseTransformLayer):
    version = '5.2.0'
    r"""Randomly rotate and shaer image / resized image by Affinetransform
    Author : Dr. David Shen
    Date : 2024/3/8
    
    Required Data's Keys:
        - img
        - img_shape
        - bbox 
        - kps

    Modified Keys:
        - img
        - bbox
        - kps
        - transform2src (option)
        - meta_info (option)
        - bbox_scale (option)

    Args:
        prob (float): The flipping probability. Defaults to 0.5

    Note: 
        it's different from mmpose affine ftansform,
        there are 3 diiffrent methods to resize image to model's input size
        -1. keep bbox apect ratio : use padding to resized image with less background
        -2. no keep bbox aspect but keep object(person)'s aspect
        -3. no keep bbox and object(person)'s aspect to directly resize bbox to model's input size

    TO DO: support batch image

    self.compute_dtype
    """

    def __init__(self,
            target_image_size_yx : Tuple[int]=(256,192),
            interpolation : str = "bilinear", 
            use_udp : bool =True,
            do_clip : bool = True,
            is_train : bool = True,
            keep_bbox_aspect_prob = 0.5,
            rotate_prob = 0.5,
            shear_prob = 0.5,
            MaxRot_deg=30., 
            MaxShear_deg=15.,
            test_mode=False ,
            **kwargs): 
        super().__init__(**kwargs)
        'base cfg'
        self.use_udp = use_udp 
        self.interpolation = interpolation
        self.do_clip = do_clip
        self.is_train = is_train
        self.test_mode = test_mode

        self.target_size_yx_int = target_image_size_yx
        self.target_size_xy = tf.reverse(
            tf.cast(target_image_size_yx,dtype=self.compute_dtype),axis=[0]
        )
        self.aspect_ratio_xy = self.target_size_xy[0]/self.target_size_xy[1]

        'prob and params'
        self.keep_bbox_aspect_prob = keep_bbox_aspect_prob if self.is_train else 1.
        self.rotate_prob = rotate_prob if self.is_train else 0.
        self.shear_prob = shear_prob if self.is_train else 0.
        self.max_rotat_deg = MaxRot_deg 
        self.max_shear_deg = MaxShear_deg
        self.deg2rad = tf.cast(0.017453292519943295, dtype=self.compute_dtype)


    def random_prob(self):
        return tf.random.uniform(())if not self.test_mode else 0.


    def get_ops(self):
        rot_op = True if self.rotate_prob > self.random_prob() else False
        shear_op = True if self.shear_prob > self.random_prob() else False
        keep_bbox_aspect = True if self.keep_bbox_aspect_prob > self.random_prob() else False
        return rot_op, shear_op, keep_bbox_aspect 

    @staticmethod
    def _resize_keep_bbox_aspect(image_size_xy, bbox_wh): 
        tf_ratio_xy = tf.cast(
            image_size_xy/bbox_wh, dtype=bbox_wh.dtype
        )
        eff_tf_ratio = tf.math.reduce_min(
            tf_ratio_xy
        )
        cond = tf.equal(
            tf_ratio_xy, eff_tf_ratio 
        )
        resize_shape_xy = tf.where(
            cond,image_size_xy,bbox_wh*eff_tf_ratio
        ) 
        return resize_shape_xy
    
        
    @staticmethod
    def _tf_xy_point_affine_transform(pt, matAffine_2x3):
        total_points = tf.shape(pt)[0]
        pt = tf.concat(
            [pt[...,:2], tf.ones(shape=(total_points,1), dtype=pt.dtype)],axis=-1) #(17,3)
        pt = tf.expand_dims(pt, axis=-1) #(17,3,1)
        matAffine_2x3 =  tf.expand_dims(matAffine_2x3, axis=0)  #(1,2,3) 
        new_pt = tf.linalg.matmul(matAffine_2x3,pt,transpose_b=False) #(17,2,1)  M_2x3 @ PT_3x1 = X_2x1
        new_pt = tf.squeeze(new_pt)  # #(17,2)
        return new_pt
    
    @staticmethod
    def _get_translation_matrix(trans_xy: float ) -> tf.Tensor:
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
            bbox_wh : Optional[tf.Tensor] = None 
    ) -> tf.Tensor:
        
        r'''
        The transform matrix looks like:
            (cos_p,   -sin_p,  x_offset)
            (sin_p,    cos_p,  y_offset)
            (    0,        0,         1)
            We flatten the matrix to `[cos_p, -sin_p, x_offset, sin_p, cos_p, y_offset, 0., 0.]` for
            use with ImageProjectiveTransformV3.
        '''
        compute_dtype = bbox_wh.dtype
        if rotate_rad==0.:
            return tf.eye(3, dtype=compute_dtype)

        #rotate_rad = rot_deg*self.deg2rad
        cos_p = tf.cast( tf.math.cos(rotate_rad), dtype=compute_dtype) #(b,1,1)
        sin_p = tf.cast( tf.math.sin(rotate_rad), dtype=compute_dtype) #(b,1,1)
        x_offset = (bbox_wh[0] - (bbox_wh[0]*cos_p - bbox_wh[1]*sin_p) )/2.
        y_offset = (bbox_wh[1] - (bbox_wh[0]*sin_p + bbox_wh[1]*cos_p) )/2.

        rotation_matrix = tf.cast(
            [
                [cos_p,   -sin_p,   x_offset],
                [sin_p,    cos_p,   y_offset],
                [0,           0,          1]
            ],
            dtype=compute_dtype
        )#[3,3]
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(
            scale_ratio: Tensor
        ) -> tf.Tensor:
        r'''
        The transform matrix looks like:
            (s, 0, 0)
            (0, s, 0)
            (0, 0, 1)
            where the last entry is implicit.

            We flatten the matrix to `[1.0, x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]` for
            use with ImageProjectiveTransformV3.
        '''

        scaling_matrix = tf.cast(
            [
                [scale_ratio[0],              0,   0],
                [0,              scale_ratio[1],   0],
                [0,                           0,   1]
            ],
            dtype=scale_ratio.dtype
        )#[3,3]
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(
            xy_shear_rad: Tensor,
            shape_xy : Tensor
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
            return tf.eye(3, dtype= compute_dtype)
        
        tan_pxy = tf.cast(
            tf.math.tan(xy_shear_rad), dtype=compute_dtype
        )#(2,)

        x_shift = - 0.5*shape_xy[1]*tan_pxy[0] 
        y_shift =  -0.5*shape_xy[0]*tan_pxy[1]

        shear_matrix = tf.cast(
            [
                [1,           tan_pxy[0],   x_shift],
                [tan_pxy[1],           1,   y_shift],
                [0,                     0,        1]
            ],
            dtype=compute_dtype
        )#[3,3]
        return shear_matrix
    def get_random_affine_matrix(
            self,
            bbox_xywh : Tensor, 
            scale_ratio : Tensor,
            rot_op_enable : bool = False, 
            shear_op_enable : bool = False,
            resize_shape_xy : Optional[Tensor]= None,                          
    ) :
        '#1'
        matAffine_3x3 = self._get_translation_matrix(
            trans_xy=bbox_xywh[:2]
        )

        '#2'
        if rot_op_enable:
            # rot_deg = tf.random.uniform(
            #         shape=(),
            #         minval = -self.max_rotat_deg,
            #         maxval = self.max_rotat_deg
            # )
            rot_deg = tf.random.truncated_normal(
                shape=(),
                mean=0.0,
                stddev=0.5,
                dtype=self.compute_dtype
            )*self.max_rotat_deg

            matAffine_3x3 = tf.linalg.matmul(
                matAffine_3x3,
                self._get_rotation_matrix(
                    rot_deg*self.deg2rad, bbox_xywh[2:]
                )
            )

        '#3 scale'
        matAffine_3x3 = tf.linalg.matmul(
            matAffine_3x3,
            self._get_scaling_matrix(scale_ratio),
        )

        '#4 shear'
        if shear_op_enable:
            # shear_xy_deg = tf.random.uniform(
            #         shape=(2,),
            #         minval = -self.max_shear_deg,
            #         maxval= self.max_shear_deg
            # )
            shear_xy_deg = tf.random.truncated_normal(
                shape=(2,),
                mean=0.0,
                stddev=0.5,
                dtype=self.compute_dtype
            )*self.max_shear_deg

            matAffine_3x3 = tf.linalg.matmul(
                matAffine_3x3,
                self._get_shear_matrix(
                    shear_xy_deg*self.deg2rad, 
                    resize_shape_xy
                ),
            )
        return matAffine_3x3

    def padding_to_target_size(
            self, tf_dst_img ,  dst_kps):

        vis = dst_kps[...,2:3]
        crop_image_shape_yx = tf.shape(tf_dst_img)[:2]
        tf_dst_img = tf.image.resize_with_crop_or_pad(
                tf_dst_img, self.target_size_yx_int[0], self.target_size_yx_int[1]
        )
        kps_offset_pad_yx =tf.cast( 
                (self.target_size_yx_int-crop_image_shape_yx)//2 ,dtype=self.compute_dtype
        )
        kps_offset_pad_xy = tf.reverse(kps_offset_pad_yx, axis=[0])
        dst_kps = tf.where(
            tf.equal(vis, 0.),
            tf.cast(0, dtype=dst_kps.dtype), 
            dst_kps[...,:2]+kps_offset_pad_xy
        )
        dst_kps = tf.concat([dst_kps,vis],axis=-1) 
        return tf_dst_img, dst_kps, kps_offset_pad_xy
    

    def transform(self, data : Dict[str,Tensor],**kwargs) -> Dict[str,Tensor]:
        
        bbox_xywh = data["bbox"]
        image = data["image"]
        kps = data["kps"]
        image = self.img_to_tensor(image)

        'determine opts'
        rot_op_enable, shear_op_enable, keep_bbox_aspect = self.get_ops()

        'obtain resize img shape'
        if keep_bbox_aspect: 
            resize_shape_xy = self._resize_keep_bbox_aspect(self.target_size_xy, bbox_xywh[2:])
        else:
            # if tf.random.uniform(())>0.5:
            #     #bbox_xywh = self.fix_aspect_ratio(bbox_xywh, self.aspect_ratio_xy)
            #     bbox_xywh =  fix_bboxes_aspect_ratio(bbox_xywh, self.aspect_ratio_xy)
            bbox_xywh =  fix_bboxes_aspect_ratio(bbox_xywh, self.aspect_ratio_xy)
            resize_shape_xy = self.target_size_xy
        
        
        bbox_wh = bbox_xywh[2:]
        if data.get('bbox_scale') is not None:
            data['bbox_scale'] = bbox_wh

        'clean kps outside bbox'
        kps = tf_clean_kps_outside_bbox(kps, bbox_xywh, sigma=2.)

        'ubias resized - scale ratio'
        if self.use_udp :
            scale_xy = tf.cast(
                bbox_wh/(resize_shape_xy-1.), dtype=self.compute_dtype
        )
        else:
            scale_xy = tf.cast(
                bbox_wh/(resize_shape_xy), dtype=self.compute_dtype
            )

        tf_matAffine_3x3 = self.get_random_affine_matrix(
            bbox_xywh,  
            scale_xy, 
            rot_op_enable, 
            shear_op_enable,
            resize_shape_xy, 
        )
        tf_trans_8x1 = tf.reshape(
            tf_matAffine_3x3, shape=(9,)
        )[:8]
        'img afine transform nearest/bilinear'
        resize_shape_yx  = tf.cast(
            tf.reverse(resize_shape_xy, axis=[0] ), dtype=tf.int32
        )
        if True :
            tf_dst_img = tfa.image.transform(image, 
                tf.cast(tf_trans_8x1, dtype=tf.float32), 
                interpolation=self.interpolation, 
                fill_mode='constant', 
                output_shape=resize_shape_yx
            ) #image_size_yx 
        else:
            #https://github.com/tensorflow/tensorflow/issues/5519
            tf_dst_img = tf.raw_ops.ImageProjectiveTransformV3(
                images = image[None,...],
                transforms = tf.cast(tf_trans_8x1[None,...], dtype=tf.float32),
                fill_value = tf.cast(0., tf.float32),
                fill_mode ="CONSTANT",
                interpolation = self.interpolation.upper(), 
                output_shape =  resize_shape_yx                           
            )
            tf_dst_img = tf_dst_img[0,...]

        'dst kps '
        tf_trans_inv = tf.linalg.inv(
            tf.cast(
                tf_matAffine_3x3, 
                dtype= kps.dtype if tf.__version__ > '2.10.0' else tf.float32
            )
        ) # (3,3)
        dst_kps = self._tf_xy_point_affine_transform(
            kps[...,:2], 
            tf.cast(tf_trans_inv[:2,:], dtype=kps.dtype)  
        )
        
        'output'
        vis = tf.expand_dims(kps[...,2],axis=-1)       #(17,)=>#(17,1)
        dst_kps = tf.concat([dst_kps, vis],axis=-1)     #(17,2)=>#(17,3)  
        
        'clip kps '
        if self.do_clip :
            dst_kps = tf_kps_clip_by_img(
                dst_kps, resize_shape_yx
            )  

        'padding img if img keeps aspect'
        if keep_bbox_aspect :
            tf_dst_img, dst_kps, kps_offset_pad_xy = self.padding_to_target_size(tf_dst_img, dst_kps)
        else:
            kps_offset_pad_xy =  tf.constant([0.,0.], dtype=self.compute_dtype)

       
        'update data'
        data = self.update_data_img(tf_dst_img, data)
        data['kps'] = dst_kps
        data['bbox'] = bbox_xywh
    
        'use to pred_transform'
        if 'transform2src' in data.keys():
            data['transform2src'] = {'scale_xy' : scale_xy,
                                    'bbox_lt_xy' : bbox_xywh[:2],
                                    'pad_offset_xy' : kps_offset_pad_xy,
            }

        if 'meta_info' in data.keys() and not self.is_train:
            #data["meta_info"]["src_image"] = data['image']
            data['meta_info']['src_num_keypoints'] = tf.math.count_nonzero(kps[:,2], axis=-1)
            data['meta_info']['src_keypoints'] = kps  #tf.reshape(kps,(-1,))
            #data['meta_info']['src_bbox'] = bbox_xywh
        return data 
