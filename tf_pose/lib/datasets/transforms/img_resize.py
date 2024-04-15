from typing import Dict, List, Optional, Tuple, Union
import tensorflow as tf
from tensorflow import Tensor
from lib.Registers import TRANSFORMS
from  .base import CVBaseTransformLayer


@TRANSFORMS.register_module()
class ImageResize(CVBaseTransformLayer):
    VESRION = '2.0.0'
    r"""
    to do : 1. support keypoint conversion
            
    
    """

    def __init__(self,
                resized_shape = (640,640),
                symmetric_padding_prob : float = 0.5, 
                **kwargs):
        super().__init__(**kwargs)
        self.symmetric_padding_prob =symmetric_padding_prob
        self.resized_shape = tf.cast(resized_shape, dtype=tf.int32)

    def transform(
            self, data : Dict[str,Tensor], **kwargs
    ) -> Dict[str,Tensor]:
        
        """Get prepared data 
        prepare data from parsed data; 
        to add some common info. used in transform pipeline
        i.e. image_size, kps_vis,....
        Args:
            data (dict): parsed data ; dict(img, kps, bbox)
        Returns:
            dict (dict): prepared data ; dict(img, kps, bbox,img_shape,kps_vis)    
        """
        self.verify_required_keys(data,['image','bbox'])
        'formatting image type'
        image = self.img_to_tensor(data["image"]) 

        #image = data['image']
        bboxes_xywh = data['bbox']

        image_shape = tf.cast(
            tf.shape(image)[:2], dtype=self.compute_dtype
        )
        tf_ratio = tf.cast(
            self.resized_shape, dtype=self.compute_dtype
        )/image_shape

        tf_ratio = tf.math.reduce_min(tf_ratio)
        resize_ratio = tf.cast(
            tf.cast(image_shape,dtype=self.compute_dtype)*tf_ratio, dtype=tf.int32
        )
        resize_img = tf.image.resize(
            image, size=resize_ratio
        )

        'Get bbox for resize_img'
        bboxes_xywh = bboxes_xywh*tf_ratio
        bboxes_offset = tf.constant([0.,0.],dtype=self.compute_dtype)
        'Get padded_resize_img'


        padded_resize_img = tf.where(
            self.symmetric_padding_prob >tf.random.uniform(()),
            tf.image.resize_with_crop_or_pad(
                resize_img,   self.resized_shape[0],  self.resized_shape[1]
            ),
            tf.image.pad_to_bounding_box(
                resize_img, 0, 0,  self.resized_shape[0],  self.resized_shape[1]
            )
        )


        if self.symmetric_padding_prob >tf.random.uniform(()) :
            symmetric_padding = True
        else:
            symmetric_padding = False

        'Get padded_resize_img'
        if symmetric_padding :
            padded_resize_img = tf.image.resize_with_crop_or_pad(
                resize_img,   self.resized_shape[0],  self.resized_shape[1]
            )
        else:
            padded_resize_img = tf.image.pad_to_bounding_box(
                resize_img, 0, 0,  self.resized_shape[0],  self.resized_shape[1]
            )
            
        if data.get('bbox',None) is not None:
            bboxes_xywh = data['bbox']
            'Get resized bbox for resize_img'
            bboxes_xywh = bboxes_xywh*tf_ratio
            if symmetric_padding:            
                bboxes_offset = tf.cast( 
                    (self.resized_shape - tf.shape(resize_img)[:2])/2 ,dtype=self.compute_dtype
                )
                bboxes_xywh = tf.stack(
                    [
                        bboxes_xywh[...,0]+bboxes_offset[1],
                        bboxes_xywh[...,1]+bboxes_offset[0],
                        bboxes_xywh[...,2],
                        bboxes_xywh[...,3]
                    ], 
                    axis=-1
                )
            data['bbox'] = bboxes_xywh

        'update img and modify its type to data'
        data = self.update_data_img(padded_resize_img, data)  
        return data    
    

#########################################################################
#
#
############################################################################
@TRANSFORMS.register_module()
class RandomPadImageResize(CVBaseTransformLayer):
    VERSION = '2.1.0'
    SUPPORT_METHODS = ['lt', 'rt','lb', 'rb', 'center']
    r""" Pad the image
    date : 2024/3/24
    author : Dr. Shen 

    image will be resized by preserve_aspect_ratio method, 
    and then using padding to achieve target_size

    Required Keys:
        - img
        - img_shape
        - bbox (optional)
        - kps (optional) 

    Required Keys:
        - img
        - img_shape
        - bbox (optional)
        - kps (optional) 

    Args:
        target_szie (int): target size of image ,
        pad_type (str) : defaults to 'lt'
        pad_val (int) :  defaults to 0
        use_udp (bool) : defaults to False
                       

    References:
        - [Inspired by 'Pad'@mmdet] (https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/transforms/transforms.py)

    Note:


    Examples:

    ```python



    """   
    def __init__(
            self, 
            target_size : Union[Tuple,List] = (640,640),
            pad_types : Union[str,List[str]] ='center',
            pad_val : int = 0,
            use_udp : bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        
        if not isinstance(target_size, (list, tuple)) or len(target_size)!=2:
            raise TypeError(
                "resize_shape must be 'tuple' or 'list' with len=2  "
                f"but got {type(target_size)} with len={len(target_size)}"
            )
        if not all([isinstance(x, int) for x in target_size]):
            raise TypeError(
                "dtype of resize_shape must be int "
                f"but got {type(target_size[0])} and {type(target_size[1])}"
            ) 
        if not isinstance(pad_val, int) or pad_val>255 or pad_val<0:
            raise TypeError(
                "dtype of pad_val must be int and  beteewn 0 and 255"
                f"but got {target_size[0]}@{type(target_size[0])} " 
                f" and {target_size[1]}@{type(target_size[1])}"
            )
        self.target_size = target_size
        self.pad_val = pad_val
        self.use_udp = use_udp
        #self.methods_map = {'lt' : 0, 'rt' : 1, 'lb' : 2, 'rb': 3, 'center' : 4}
        #self.supported_methods = ['lt', 'rt','lb', 'rb', 'center']

        if type(pad_types)==str:
            pad_types = [pad_types] 
        
        if not all( [ pad_type in self.SUPPORT_METHODS for pad_type in pad_types]):
            raise ValueError(
                "item in pad_types must be in ['lt', 'rt','lb', 'rb', 'center'] "
                f"but got pad_types : {pad_types}"
            )
        
        self.pad_methods  = pad_types


    def _lt(self, padding_x, padding_y):
        paddings = tf.cast([[0,padding_y],[0,padding_x],[0,0]],dtype=tf.int32)
        offset_xy = tf.cast([0.,0.], dtype=self.compute_dtype)
        return paddings, offset_xy
    
    def _rt(self, padding_x, padding_y) :   
        paddings = tf.cast([[0,padding_y],[padding_x,0],[0,0]],dtype=tf.int32)
        offset_xy = tf.cast([padding_x,0], dtype=self.compute_dtype)
        return paddings, offset_xy
    
    def _lb(self, padding_x, padding_y) :   
        paddings = tf.cast([[padding_y,0],[0,padding_x],[0,0]],dtype=tf.int32)
        offset_xy = tf.cast([0,padding_y], dtype=self.compute_dtype)
        return paddings, offset_xy
    
    def _rb(self, padding_x, padding_y):
        paddings = tf.cast([[padding_y,0],[padding_x,0],[0,0]],dtype=tf.int32)
        offset_xy = tf.cast([padding_x,padding_y], dtype=self.compute_dtype)
        return paddings, offset_xy
    
    def _center(self, padding_x, padding_y):
        half_padding_x = padding_x//2
        half_padding_y = padding_y//2
        paddings = tf.cast(
            [
                [half_padding_y, padding_y-half_padding_y],
                [ half_padding_x, padding_x-half_padding_x],
                [0,0]
            ]
            ,dtype=tf.int32
        )
        offset_xy = tf.cast([half_padding_x,half_padding_y], dtype=self.compute_dtype)
        return paddings, offset_xy  
    
    def _rand_method(self, methods) :     
        sel_method = tf.gather(
            methods, 
            tf.random.uniform(
                shape=(), 
                maxval=len(methods), 
                dtype=tf.int32
            )
        )
        return sel_method
    
    def transform(
            self,  data : Dict[str,Tensor],  pad_type : Optional[str]= None, **kwargs
    ) -> Dict[str,Tensor]:  

        if isinstance(pad_type, (tf.Tensor, str)):
            sel_method = tf.convert_to_tensor(pad_type) #pad_type
            tf.debugging.assert_equal(
                tf.reduce_any( 
                    tf.equal(
                        sel_method[None],  
                        tf.cast(self.SUPPORT_METHODS, dtype=tf.string)[None,:]
                    )
                ),
                True ,
                "sel_method in pad_types must be in ['lt', 'rt','lb', 'rb', 'center']"          
            )
        else:
            sel_method = self._rand_method(
                self.pad_methods
            )        

        'formatting image type'
        image = self.img_to_tensor(data["image"])  
        src_img_shape = tf.shape(image)[:2]

        image = tf.image.resize(
            image, 
            size=self.target_size, 
            preserve_aspect_ratio=True
        )
        resized_img_shape = tf.shape(image)[:2]

        if self.use_udp :
            resize_ratio = tf.cast( 
                (resized_img_shape[0]-1)/(src_img_shape[0]-1), dtype=self.compute_dtype
            )
        else:
            resize_ratio = tf.cast( 
                resized_img_shape[0]/src_img_shape[0], dtype=self.compute_dtype
            )

        padding_y = self.target_size[0]-resized_img_shape[0] #int
        padding_x = self.target_size[1]-resized_img_shape[1]


        paddings, offset_xy = tf.case(
            [
                (tf.equal(sel_method, 'lt'), lambda :self._lt(padding_x, padding_y)), 
                (tf.equal(sel_method, 'rt'), lambda :self._rt(padding_x, padding_y)),
                (tf.equal(sel_method, 'lb'), lambda :self._lb(padding_x, padding_y)),
                (tf.equal(sel_method, 'rb'), lambda :self._rb(padding_x, padding_y)),
                (tf.equal(sel_method, 'center'), lambda :self._center(padding_x, padding_y))
            ],
            default = lambda :self._center(padding_x, padding_y)
        )
        image = tf.pad(
            image,  paddings, mode='CONSTANT', constant_values=self.pad_val
        )

        'update img and modify its type to data'
        data = self.update_data_img(image, data)  #tf.tensor->tf.ragged_tensor or tf.tensor->tf.tensor
        if data.get('image_size', None) is not None:
            data['image_size'] = tf.shape(image)[:2]

        if data.get('bbox', None) is not None:
            'mask'
            bboxes_xywh = data["bbox"] #bboxes_xywh : (n,4) or (4,)
            mask = tf.less_equal( 
                x = tf.math.reduce_max(bboxes_xywh, axis=-1), y = 0.
            )
            bboxes_xywh = bboxes_xywh*resize_ratio
            bboxes_xywh = tf.concat(
                [bboxes_xywh[...,:2]+offset_xy, bboxes_xywh[...,2:]], axis=-1
            )
            bboxes_xywh = tf.where(
                mask[...,None], 
                tf.cast(0., dtype=self.compute_dtype), 
                bboxes_xywh
            )
            data["bbox"] = bboxes_xywh

        if data.get('kps', None) is not None:
            'mask'
            kps = data["kps"] #kps : (17,3)
            vis = kps[...,2:3]
            
            kps_xy = tf.where(
                tf.equal(vis, 0.),  
                tf.zeros_like(kps[...,:2]), 
                kps[...,:2]*resize_ratio+offset_xy
            )
            
            kps = tf.concat( 
                [kps_xy, vis],axis=-1
            )
            data["kps"] = kps
     
        return data
