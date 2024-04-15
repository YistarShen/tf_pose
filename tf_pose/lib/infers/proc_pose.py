
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from tensorflow import Tensor
import tensorflow as tf
from tensorflow.keras.layers import  Input
from tensorflow.keras.models import Model
import os 
from .base import BaseProc
from lib.Registers import INFER_PROC

def set_thr_coco_style(face_thr : float=0.75, 
                    body_thr  : float=0.25, 
                    hand_foot_thr : Optional[float]=None, 
                    num_kps : int=17):
    
    if face_thr is None :
        face_thr = 0.5   
    face_thr = tf.ones(shape=(5,), dtype=tf.float32)*face_thr

    if body_thr is None :
        body_thr = 0.5
    body_thr = tf.ones(shape=(12,), dtype=tf.float32)*body_thr

    if num_kps>17:
        hand_foot_thr =  0.25 if hand_foot_thr is None else hand_foot_thr
        hand_foot_thr = tf.ones(shape=(num_kps-17,), dtype=tf.float32)*hand_foot_thr
        val_thr = tf.concat([face_thr, body_thr, hand_foot_thr], axis=0)
    else:
        val_thr = tf.concat([face_thr, body_thr], axis=0)

    assert val_thr.shape[0]==num_kps
    return val_thr




@INFER_PROC.register_module()
class PreProcess_Pose(BaseProc):
    version = '1.0.0'
    r""" PreProcess_Pose

    """
    def __init__(self, 
                 img_shape_yx : Tuple[int]=(256,192), 
                 batch_size :int = 1,
                 keep_aspect = False,
                 use_udp = True,
                 name=None ):
        
        self.img_shape_yx = img_shape_yx
        self.keep_aspect = keep_aspect
        self.use_udp = use_udp

        if 0 :
            print("\n\n ======= PoseDet Model CFG  -----< PreProcess > =============")
            print(f"Model input shape : {self.img_shape_yx} \n\n")
            print(f"use_udp  : {self.use_udp} \n\n")
            print(f"keep_aspect  : {self.keep_aspect} \n\n")

        super().__init__(name=name,  
                        batch_size = batch_size)
        
    def Set_InputsTensorSpec(self, batch_size): 
        """Set_InputsTensorSpec.
        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """
        #self.inputs = [ Input(shape=TensorSpec, batch_size=self.batch_size) for TensorSpec in InputsTensorSpec ]
        InputsTensor = [ Input(shape=(None,None,3), batch_size=batch_size)]
        return InputsTensor 

    def bbox_yx_to_xywh(self, bbox_yx):
        '''
        if(bbox_yx[...,0]>bbox_yx[...,2] or bbox_yx[...,1]>bbox_yx[...,3]):
          raise ValueError("bbox_yx[0]<bbox_yx[2] and bbox_yx[1]<bbox_yx[3]") 
        '''
        y = bbox_yx[0]
        x = bbox_yx[1]
        w = bbox_yx[3] - bbox_yx[1]
        h = bbox_yx[2] - bbox_yx[0]
        #bbox_xywh = tf.stack([x,y,w,h],axis=-1)
        return tf.cast(x, dtype=tf.int32), tf.cast(y, dtype=tf.int32), tf.cast(w, dtype=tf.int32), tf.cast(h, dtype=tf.int32)  
        
    def tf_img_norm_transform(self, img):
        img_mean = tf.constant([0.485, 0.456, 0.406],dtype=tf.float32)
        img_std = tf.constant([0.229, 0.224, 0.225],dtype=tf.float32)
        img = img / 255.0
        img = (img - img_mean)/img_std
        return img
    
    def Set_InputsTensorSpec(self, batch_size): 
        """Set_InputsTensorSpec.
        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        Note:
            two inputs, one is iamge and other is bbox_yx
        """
    
        InputsTensor = [ Input(shape=(None,None,3), batch_size=batch_size), Input(shape=(4,) ,batch_size=batch_size)]
        return InputsTensor
    
    def forward(self, 
                x : Union[List[Tensor],Tuple[Tensor]]) -> Union[List[Tensor],Tuple[Tensor]]:
        
        assert isinstance(x,List),"unput format must be list | Tuple  @PreProcess_Pose.forward"
        """
        image : (1, 256, 192, 3)
        bbox_yx : (1, 4) @float
        """
        image, bbox_yx = x

        'ensure image is int type'
        #assert bbox_yx.shape[0]== 1, "batch size of bbox_yx must be one"
        #assert image.shape[0]== 1, "batch size of image must be one"

        image = tf.cast(image, dtype=tf.int32)

        src_bbox_hw = bbox_yx[:,2:] - bbox_yx[:,:2] #(b,2)
        src_bbox_yx = bbox_yx[:, :2]  #(1,2) 
        bbox_yx = bbox_yx[0, :]
        x, y, w, h = self.bbox_yx_to_xywh(bbox_yx)
        
        'crop_img'
        crop_image = tf.image.crop_to_bounding_box(image, y, x, h, w) #(1,None, None, 3)
        
        #crop_image = image
        #crop_image = tf.slice(image, begin=[0,y,x,0], size=[1,h,w,3])
        #crop_image_shape = tf.shape(crop_image)[1:3]    
        if self.keep_aspect :
            '---------------only support batch = 1--------------------------------------------'
            'crop_resize_img crop_resized_img'
            crop_resize_img = tf.image.resize(crop_image, size=self.img_shape_yx, preserve_aspect_ratio=True)
            crop_resize_shape = tf.cast( tf.shape(crop_resize_img)[1:3], dtype=tf.float32)
            'crop_resized_pad_img '
            crop_resized_img = tf.image.resize_with_crop_or_pad(crop_resize_img, self.img_shape_yx[0], self.img_shape_yx[1])
            'crop_resized_pad_normalize_img @input must be'
            #crop_resized_norm_img = self.tf_img_norm_transform(crop_resized_pad_img) 
            if self.use_udp :
                aspect_resize2src_yx  = crop_resize_shape/src_bbox_hw #(b,2)
            else:
                aspect_resize2src_yx  = (crop_resize_shape-1.)/src_bbox_hw #(b,2)
            'offset_yx(padding size)'
            offset_padding_yx = tf.cast( tf.shape(crop_resized_img)[1:3], dtype=tf.float32) - crop_resize_shape
            offset_padding_yx = tf.expand_dims(offset_padding_yx/2 , axis=0) #(2, )
            'meta'
            meta_yx = tf.stack([aspect_resize2src_yx, offset_padding_yx, src_bbox_yx], axis=1)#(1,3,2)
            meta_xy = tf.reverse(meta_yx, axis=[2])#(1,3,2)

        else:
            '---------------only support batch = 1--------------------------------------------'
            'crop_resized_img without'
            crop_resized_img = tf.image.resize(crop_image, size=(self.img_shape_yx[0], self.img_shape_yx[1]))
            'crop_resized_pad_normalize_img @input must be'
            #crop_resized_norm_img = self.tf_img_norm_transform(crop_resized_img) 
            'aspect_resize2src_yx' 
            if self.use_udp :
                aspect_resize2src_yx  = tf.cast(self.img_shape_yx, dtype=tf.float32)/src_bbox_hw #(b,2)
            else:
                aspect_resize2src_yx  = (tf.cast(self.img_shape_yx, dtype=tf.float32)-1.)/src_bbox_hw #(b,2)
            'meta'
            offset_padding_yx = tf.zeros_like(src_bbox_hw) #(b,2)
            meta_yx = tf.stack([aspect_resize2src_yx, offset_padding_yx, src_bbox_yx], axis=1)#(1,3,2)
            meta_xy = tf.reverse(meta_yx, axis=[2])
        """
        crop_resized_pad_norm_img : (1, 256, 192, 3) @float
        crop_resized_shape_yx : (2,) @int
        bbox_xywh : (4,)  @float
        """      
        return crop_resized_img, meta_xy
    
    @tf.function(jit_compile=False)
    def __call__(self, 
                image: List[Tensor],
                bbox_yx : Tensor) ->Tuple[Tensor]:
        '''
        assert feats_list[0].shape.rank==5 and len(feats_list)==3, \
        "shape.rank of feats must be 5 @PostProcess_YOLO_Det.__call__"

        assert meta.shape.rank==2, \
        "shape.rank of meta must be 2 @PostProcess_YOLO_Det.__call__"
        '''

        src_img = tf.cast(image, dtype=tf.float32)
        bbox_yx = tf.cast(bbox_yx, dtype=tf.float32)
        
        
        assert src_img.shape.rank==3 or src_img.shape.rank==4,\
        f"shape.rank of input data must be 3 or 4, but got {src_img.shape.rank} @{self.__call__.__name__}"
        if src_img.shape.rank==3:
            src_img = tf.expand_dims(src_img,axis=0)
        
        if bbox_yx.shape.rank==1 :
            bbox_yx = tf.expand_dims(bbox_yx,axis=0)
        assert bbox_yx.shape[-1]==4 and bbox_yx.shape.rank==2, \
        f"bbox_yx.shape must be (1,4), but got {bbox_yx.shape} @{self.__call__.__name__}"  

        return self.model([src_img,bbox_yx])   

#--------------------------------------------------------------------------------------------
#
#
#--------------------------------------------------------------------------------------------
@INFER_PROC.register_module()
class PostProcess_HM_Pose(BaseProc):
    VERSION = '1.0.0'
    r""" PostProcess_HM_Pose
    
    """
    def __init__(self, 
                 hm_shape_yx :Tuple[int] = (64, 48), 
                 img_shape_yx : Tuple[int] = (256,192),
                 batch_size : int = 1,
                 num_kps : int = 17, 
                 nms_kernel : int = 3, 
                 hm_thr : Union[tf.Tensor, List[float]] = None,
                 use_udp : bool = True,
                 src_img_frame : bool = True,
                 name=None):

        'predictions post process (heatmap-decoder)'
        self.use_udp = use_udp
        self.hm_shape_yx = hm_shape_yx
        #self.hm_height = tf.cast(hm_shape_yx[0],dtype=tf.float32)
        #self.hm_width = tf.cast(hm_shape_yx[1],dtype=tf.float32)
        self.num_kps_points = num_kps

        self.nms_kernel = nms_kernel
        self.src_img_frame = src_img_frame

        if self.use_udp :
            scale_yx_hm2img = ( tf.cast(img_shape_yx, dtype=tf.float32)-1.)/(tf.cast(hm_shape_yx, dtype=tf.float32)-1.)
        else:
            scale_yx_hm2img = ( tf.cast(img_shape_yx, dtype=tf.float32))/(tf.cast(hm_shape_yx, dtype=tf.float32))
        self.scale_xy_hm2img = tf.reverse(scale_yx_hm2img, axis=[0])
        #self.hm2img_scalar = udp_scale_xy_hm2img


        'keypoint hr'
        'set hm_thr to decode'
        if hm_thr is None :
            self.hm_thr = tf.zeros(shape=(num_kps,), dtype=tf.float32)
        else:
            if isinstance(hm_thr,tf.Tensor): 
                self.hm_thr = hm_thr
            elif isinstance(hm_thr,List): 
                self.hm_thr = tf.constant(hm_thr, dtype=tf.float32)
            else:
                raise RuntimeError(f"type of hm_thr must be tf.Tensor|List,but got {type(hm_thr)}") 

            assert  self.hm_thr.shape[0] == num_kps, \
             f"hm_thr.shape[0] must be equal to {num_kps}, but got {self.hm_thr.shape[0]} @PostProcess_HM_Pose"  

        if 0 :
            print("\n\n ========= PoseDet Model CFG  -----< PostProcess > ======================")
            print(f"num_kps : {self.num_kps_points}")
            print(f"heatmap_thr : {self.hm_thr}")
            print(f"nms_kernel : {self.nms_kernel}")
            print(f"heatmap_shape : {hm_shape_yx}")
            print(f"hm2img_scalar : { self.scale_xy_hm2img}")

        super().__init__(name=name,  
                        batch_size = batch_size)
        
    def _nms(self, heat, kernel=3):
        hmax = tf.nn.max_pool2d(heat, kernel, 1, padding='SAME')
        keep = tf.cast(tf.equal(heat, hmax), tf.float32)
        return heat*keep
    
    def Set_InputsTensorSpec(self, batch_size): 
        """Set_InputsTensorSpec.
        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        Note:
            two inputs, one is iamge and other is bbox_yx
        """
        hm = Input(shape=(*self.hm_shape_yx, self.num_kps_points), batch_size=batch_size)
        meta = Input(shape=(3,2), batch_size=batch_size)

        InputsTensor = [ hm, meta ]
        return InputsTensor
    

    def forward(self, 
                x : Union[List[Tensor],Tuple[Tensor]]) -> Union[List[Tensor],Tuple[Tensor]]:
        
        'format input x'
        batch_heatmaps, meta_xy = x

        assert batch_heatmaps.shape[-1]==self.num_kps_points, \
        f"num_kps={batch_heatmaps.shape[-1]} in heatmap doesn't match setting={self.num_kps_points} @PostProcess_HM_Pose.forward"


        batch_heatmaps = tf.cast(batch_heatmaps, dtype=tf.float32)
        batch, hm_height, hm_width, num_point = batch_heatmaps.get_shape()
        batch_heatmaps = self._nms(batch_heatmaps) #(1,96,72,17)
        flatten_tensor = tf.reshape(batch_heatmaps, (-1, hm_height * hm_width, num_point)) #(1, 96*72, 17)
 
        'Argmax of the flat tensor'
        argmax = tf.argmax(flatten_tensor, axis=1) #(1,17)
        argmax = tf.cast(argmax, tf.int32) #(1,17)
        scores = tf.math.reduce_max(flatten_tensor, axis=1) #(1,17)
        'Convert 1D indexes into 2D coordinates'
        argmax_y = tf.cast( (argmax//hm_width), tf.float32)  #(1,17)
        argmax_x = tf.cast( (argmax%hm_width), tf.float32)   #(1,17)
        '''
        if self.kps_normalize:
          argmax_x = argmax_x / tf.cast(hm_width, tf.float32)
          argmax_y = argmax_y / tf.cast(hm_height, tf.float32)
        '''
        batch_keypoints = tf.stack((argmax_x, argmax_y, scores), axis=2)    # Shape: batch * 3 * n_points  (1,17,3)
        mask = tf.greater(batch_keypoints[:, :, 2], self.hm_thr ) #(1,17)

        #kps_x = tf.where(mask, batch_keypoints[:, :, 0], 0)*self.hm2img_scalar #(1,17)
        #kps_y = tf.where(mask, batch_keypoints[:, :, 1], 0)*self.hm2img_scalar #(1,17)

        kps_xy = tf.where(mask[:,:,None], batch_keypoints[:, :, :2], 0)*self.scale_xy_hm2img


        if self.src_img_frame :
            'meta'
            aspect_resize2src_xy = meta_xy[:,0,:] #(1,2)
            offset_padding_xy = meta_xy[:,1,:] #(1,2)
            src_bbox_xy =  meta_xy[:,2,:]  #(1,2)

            'trasform to src img frame'
            kps_xy = (kps_xy-offset_padding_xy[:,None,:])/aspect_resize2src_xy[:,None,:] + src_bbox_xy[:,None,:] #(1, 17,2) 
            kps_xy = tf.where(mask[...,None], kps_xy, 0.) #(1)
            #kps_x = kps_yx[...,1]
            #kps_y = kps_yx[...,0]

        #batch_keypoints = tf.stack((kps_yx[...,1], kps_yx[...,0], scores), axis=2)

        batch_keypoints = tf.concat([kps_xy, scores[:,:,None]], axis=-1) #(b,17,3)
        """
        batch_keypoints : (1, 27, 3) @float
        """
        return batch_keypoints
       
    @tf.function(jit_compile=True)
    def __call__(self, 
                heatmaps: List[Tensor],
                meta_xy : Tensor) ->Tuple[Tensor]:
        
    
        assert heatmaps.shape[1:] ==(*self.hm_shape_yx, self.num_kps_points), \
        f"heatmaps.shape of feats must be {(*self.hm_shape_yx, self.num_kps_points)} @PostProcess_YOLO_Det.__call__"
    

        '''
        assert meta.shape.rank==2, \
        "shape.rank of meta must be 2 @PostProcess_YOLO_Det.__call__"
        '''

        return self.model([heatmaps, meta_xy])   
    


#--------------------------------------------------------------------------------------------
#
#
#--------------------------------------------------------------------------------------------
@INFER_PROC.register_module()
class PostProcess_SIMCC_Pose(BaseProc):
    VERSION = '1.0.0'
    r"""
    
    """
    def __init__(self, 
                simcc_coord_dims_yx : Tuple[int] =(512, 384),
                img_shape_yx :Tuple[int] = (256,192),
                batch_size : int = 1,
                num_kps : int = 17, 
                val_thr : Dict[str, float] = {"face_thr":0.5,"body_thr":0.25},
                use_udp : bool = True,
                src_img_frame : bool = True,
                name :Optional[str]=None ):
        
 
        self.num_kps = num_kps
        self.src_img_frame = src_img_frame
        self.use_udp = use_udp

        if self.use_udp :
            udp_scale_yx_simcc2img = ( tf.cast(simcc_coord_dims_yx, dtype=tf.float32)-1.)/(tf.cast(img_shape_yx, dtype=tf.float32)-1.)
            self.simcc2img_scale_xy = tf.reverse(udp_scale_yx_simcc2img, axis=[0])
        else:
            self.simcc2img_scale_xy = (tf.cast(simcc_coord_dims_yx, dtype=tf.float32))/(tf.cast(img_shape_yx, dtype=tf.float32))


        'keypoint hr to decode for simcc'
        '''
        def set_thr_coco_style(face_thr : float=0.75, 
                        body_thr  : float=0.25, 
                        hand_foot_thr : Optional[float]=None, 
        '''
        val_thr['num_kps'] = self.num_kps
        self.simcc_thr = set_thr_coco_style(face_thr = val_thr["face_thr"], 
                                            body_thr = val_thr["body_thr"],
                                            hand_foot_thr = None,
                                            num_kps=self.num_kps)

        '''
        if val_thr is None :
            self.simcc_thr = tf.zeros(shape=(num_kps,), dtype=tf.float32)
        else:
            if isinstance(val_thr,tf.Tensor): 
                self.simcc_thr = val_thr
            elif isinstance(val_thr,List): 
                self.simcc_thr = tf.constant(val_thr, dtype=tf.float32)
            else:
                raise RuntimeError(f"type of hm_thr must be tf.Tensor|List,but got {type(val_thr)}") 

            assert  self.simcc_thr.shape[0] == num_kps, \
             f"hm_thr.shape[0] must be equal to {num_kps}, but got {self.simcc_thr.shape[0]} @PostProcess_SIMCC_Pose"  
        '''

        self.simcc_size_splits_xy = simcc_coord_dims_yx[::-1]
        self.simcc_coord_dims_sum =  sum(self.simcc_size_splits_xy)
        print("\n\n ========= PoseDet Model CFG  -----< PostProcess > ======================")
        print(f"num_kps : {self.num_kps}")
        print(f"simcc_thr : {self.simcc_thr}")
        print(f"simcc_coord_dims_xy : {simcc_coord_dims_yx[::-1]}")
        print(f"simcc2img_scale_xy : { self.simcc2img_scale_xy}")
        #print(f"kps_normalize : { self.kps_normalize} \n\n")

        
        super().__init__(name=name,  
                        batch_size = batch_size)
        
    def Set_InputsTensorSpec(self, batch_size): 
        """Set_InputsTensorSpec.
        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        Note:
            two inputs, one is iamge and other is bbox_yx
        """
        feat = Input(shape=(self.num_kps, self.simcc_coord_dims_sum), batch_size=batch_size)
        meta = Input(shape=(3,2), batch_size=batch_size)
        InputsTensor = [feat, meta ]
        return InputsTensor
    
    def forward(self, 
                 x : Union[List[Tensor],Tuple[Tensor]]) -> Union[List[Tensor],Tuple[Tensor]]:
        """
        batch_heatmaps : (1,17,coord_x+coord_y) @float
        """

        batch_simcc_labels, meta_xy = x


        '-----simcc decode----'
        assert batch_simcc_labels.shape[1]==self.num_kps, \
        f"num_kps={batch_simcc_labels.shape[1]} in simcc didn't match kps setting={self.num_kps} @PostProcess_SIMCC_Pose.forward"

        coord_pred_x, coord_pred_y = tf.split(batch_simcc_labels, self.simcc_size_splits_xy, axis=-1)
  
        locs_x = tf.argmax(coord_pred_x, axis=-1) #(1,17)
        locs_y = tf.argmax(coord_pred_y, axis=-1) #(1,17)
        locs_xy = tf.cast(tf.stack([locs_x,locs_y], axis=-1), dtype=tf.float32)/self.simcc2img_scale_xy
        

        max_val_x  = tf.math.reduce_max(tf.nn.softmax(coord_pred_x), axis=-1)  #(1,17)
        max_val_y  = tf.math.reduce_max(tf.nn.softmax(coord_pred_y), axis=-1)  #(1,17)
        val = tf.math.minimum(max_val_x,max_val_y) #(1,17)

        mask =  tf.expand_dims(tf.greater(val, self.simcc_thr), axis=-1) #(1,17,1)
        socre = tf.cast(mask, dtype=tf.float32)

        kps_xy = tf.where(mask, locs_xy, 0.) #(1,17,2)

        if self.src_img_frame :
            'meta'
            aspect_resize2src_xy = meta_xy[:,0,:] #(1,2)
            offset_padding_xy = meta_xy[:,1,:] #(1,2)
            src_bbox_xy =  meta_xy[:,2,:]  #(1,2)
            'trasform to src img frame'
            kps_xy = (kps_xy-offset_padding_xy[:,None,:])/aspect_resize2src_xy[:,None,:] + src_bbox_xy[:,None,:] #(1, 17,2) 
            kps_xy = tf.where(mask, kps_xy, 0.) #(1)

        #batch_keypoints = tf.concat([kps_xy, val[...,None]], axis=-1) #(b,17,3)
        batch_keypoints = tf.concat([kps_xy, socre], axis=-1) #(b,17,3)
        return batch_keypoints
    

    @tf.function(jit_compile=True)
    def __call__(self, 
                simcc_feat: tf.Tensor,
                meta_xy : tf.Tensor) -> tf.Tensor:
        
        assert simcc_feat.shape[1:] ==( self.num_kps, self.simcc_coord_dims_sum), \
        f"simcc_feat.shape must be {(*self.num_kps, self.simcc_coord_dims_sum)} @PostProcess_SIMCC_Pose.__call__"

        '''
        assert meta.shape.rank==2, \
        "shape.rank of meta must be 2 @PostProcess_YOLO_Det.__call__"
        '''
        #single_person_bbox_yx, decoded_detections = self.model([feats_list, meta]) 
              
        return self.model([simcc_feat, meta_xy])   
    


#--------------------------------------------------------------------------------------------
#
#
#--------------------------------------------------------------------------------------------
@INFER_PROC.register_module()
class SmoothFilter_Pose2D(BaseProc):
    VERSION = '1.0.0'
    r"""SmoothFilter_Pose2D
    SmoothFilte is based on OneEuroFilter
    support built-in buffer to implement one input and one output
    
    """
    def __init__(self, 
                num_joints : int = 17, 
                batch_size : int = 1,
                t_e : float = 1.0 ,    
                min_cutoff : float =1.7,
                beta : float = 0.3,
                fps : int = 50, 
                xla : bool = True,
                name :Optional[str]=None ):
        
        self.t_e = t_e
        self.pi = 3.14159265359
        self.coef = tf.constant([[min_cutoff,beta,fps]], dtype=tf.float32) #(1,3)
        self.num_joints = num_joints

        'built-in buffer'
        self.x_prev = tf.Variable(tf.zeros(shape=(batch_size, num_joints, 2)), dtype=tf.float32)
        self.dx_prev = tf.Variable(tf.zeros(shape=(batch_size, num_joints, 2)), dtype=tf.float32)

        'jit_compile'
        self.jit_compile = xla

        super().__init__(name=name,  
                        batch_size = batch_size)
        
    def Set_InputsTensorSpec(self, 
                             batch_size : int = 1): 
        """Set_InputsTensorSpec.
        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        Note:
            two inputs, one is iamge and other is bbox_yx
        """

        x_curr = Input(shape=(self.num_joints ,3), batch_size=batch_size)
        x_prev = Input(shape=(self.num_joints ,2), batch_size=batch_size)
        dx_prev = Input(shape=(self.num_joints ,2), batch_size=batch_size)
        coef = Input(shape=(3,), batch_size=batch_size)
        InputsTensor = [x_curr, x_prev, dx_prev, coef]
        return InputsTensor
    
    def smoothing_factor(self, t_e, cutoff):
        r = 2 *  self.pi * cutoff * t_e
        return r / (r + 1)   

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev
        
    def forward(self, 
                x : Union[List[Tensor],Tuple[Tensor]]) -> Union[List[Tensor],Tuple[Tensor]]:
        """
        x_vis : (1,17,3)
        x_prev : (1,17,2)
        dx_prev  : (1,17,2)
        min_cutoff  : (1,17,2)
        beta  : (1,17,2)
        d_cutoff  : (1,17,2)
        coef :(1,3) [min_cutoff, beta,  fps]
        """
        x_vis, x_prev, dx_prev, coef = x

        fps = coef[0,2]

        x = x_vis[...,:2]
        vis = x_vis[...,2:3]
        t_e = tf.ones_like(x_prev)*self.t_e

        temp = tf.ones_like(x_prev)[:,:,:,None]*coef[:,None,None,:]
        min_cutoff, beta, d_cutoff = tf.unstack(temp,  axis=-1)
        'missing keypoints mask'"h36m_2d_HPE_eval copy.ipynb"
        mask = tf.greater(x, 0.)
        'The filtered derivative of the signal.'
        d_cutoff = tf.ones_like(min_cutoff)*fps
        a_d = self.smoothing_factor(t_e /fps, d_cutoff)
        dx = (x - x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, dx_prev)

        'The filtered signal'
        cutoff = min_cutoff + beta * tf.math.abs(dx_hat)
        a = self.smoothing_factor(t_e / fps, cutoff)
        x_hat = self.exponential_smoothing(a, x, x_prev)

        'missing keypoints remove'
        x_hat = tf.where(mask, x_hat,-10) 
        dx_prev = dx_hat
        return tf.concat([x_hat, vis], axis=-1),  dx_prev
    
    
    @tf.function(jit_compile=True)
    def __call__(self, 
                x_curr: tf.Tensor) -> tf.Tensor:
        
        #assert x_curr.shape.rank==3,
        
        'Inference  --------------------------------START'
        filtered_x_curr , dx_prev = self.model([x_curr, self.x_prev, self.dx_prev, self.coef]) 
        self.dx_prev.assign(dx_prev)
        self.x_prev.assign(filtered_x_curr[...,:2])
        'Inference  --------------------------------END'
        return  filtered_x_curr
        
 