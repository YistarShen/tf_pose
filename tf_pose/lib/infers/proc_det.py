
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from tensorflow import Tensor
import tensorflow as tf
from tensorflow.keras.layers import  Input
from tensorflow.keras.models import Model
import os 
from .base import BaseProc
from lib.Registers import INFER_PROC

#--------------------------------------------------------------------------------------------
#
#--------------------------------------------------------------------------------------------
@INFER_PROC.register_module()
class PreProcess_Det(BaseProc):
    def __init__(self,       
                img_shape_yx : Tuple[int] =(640,640), 
                img_expand_batch_dim : bool = True,
                batch_size : int = 1,
                padding_type : str = "corner", 
                save_model : bool = False,
                saved_model_dir : Optional[str] = None, 
                name : str = 'Det_ImgPreProc_Model' ):
        
        self.batch_size = batch_size
        self.img_shape_yx = img_shape_yx
        self.padding_method = padding_type
        self.img_expand_batch_dim = img_expand_batch_dim

        super().__init__(name=name,  
                        batch_size = batch_size)

    def tf_img_norm_transform(self, img, inv=False):
        img_mean = tf.constant([0.485, 0.456, 0.406],dtype=tf.float32)
        img_std = tf.constant([0.229, 0.224, 0.225],dtype=tf.float32)
        if (inv==False):
            img = img / 255.0
            img = (img - img_mean)/img_std
        else:
            img =  img*img_std + img_mean
        return img
    
    def Set_InputsTensorSpec(self, batch_size): 
        """Set_InputsTensorSpec.
        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """
        #self.inputs = [ Input(shape=TensorSpec, batch_size=self.batch_size) for TensorSpec in InputsTensorSpec ]
        InputsTensor = [ Input(shape=(None,None,3), batch_size=batch_size)]
        return InputsTensor
    
    def forward(self, x : Union[List[Tensor],Tuple[Tensor]]) -> Union[List[Tensor],Tuple[Tensor]]:
        """Forward function.
        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """
        input_img, = x
        # assert input_img.shape.rank==4,\
        # f"shape.rank of input_img must be 4, vut got {input_img.shape.rank} @PreProcess_Det.forward"

        'img resize to meet sellected model spec'
        tf_ratio = tf.cast(self.img_shape_yx/tf.shape(input_img)[1:3], dtype=tf.float32)
        aspect_resize2src = tf.math.reduce_min(tf_ratio)
        resize_shpae = tf.cast(tf.cast(tf.shape(input_img)[1:3],dtype=tf.float32)*aspect_resize2src, dtype=tf.int32)  # resize_ratio = (640, 360)   
        resized_image = tf.image.resize(input_img, size=resize_shpae, method='nearest') #nearest/

        'padding it should be faster(need to verify it)'
        if self.padding_method == "center":
            'it should be higher accuracy'
            out_img = tf.image.resize_with_crop_or_pad(resized_image, self.img_shape_yx[0], self.img_shape_yx[1])         
        elif self.padding_method == "corner":
            'it should be faster(need to verify it)'
            out_img = tf.image.pad_to_bounding_box(resized_image, 0, 0, self.img_shape_yx[0], self.img_shape_yx[1])  # tensor : (640, 640, 3), dtype=uint8
        else:
            raise ValueError("padding_method must be 'center' or 'corner' !!!!!@PD_Preprocess") 
        
        'offset_yx(padding size)'
        offset_yx = tf.cast( (tf.shape(out_img)[1:3] - tf.shape(resized_image)[1:3]), dtype=tf.float32)
        offset_yx = offset_yx/2 if self.padding_method == "center" else tf.constant([0., 0.], dtype=tf.float32)
        offset_yx = tf.expand_dims(offset_yx,axis=0) #(1,2)  
        
        'image normalization'
        #norm_image =  self.tf_img_norm_transform(tf.cast(out_img,dtype=tf.float32), inv=False)
        'Add Batch axis to meet model spec'
        resize_shape_yx = tf.expand_dims(tf.shape(resized_image)[1:3], axis=0)
        resize_shape_yx = tf.cast(resize_shape_yx, dtype=tf.float32)  

        src_img_shape = tf.cast( tf.shape(input_img)[1:3], dtype=tf.float32)
        meta = tf.concat([[[aspect_resize2src]],offset_yx, src_img_shape[None,...]], axis=-1, name='meta_out') #(1, 3)  

        img = tf.keras.layers.Layer(name="resized_img_out")(tf.cast(out_img,dtype=tf.float32)) 
        meta = tf.keras.layers.Layer(name="meta_out")(meta)      
            
        return img, meta 
    
    @tf.function(jit_compile=True, reduce_retracing=False)
    def __call__(self, 
                data: Tensor) -> Union[Tensor,Tuple[Tensor]] :
        """
        it's callable function not tf-model for quicklly test , input is a tensors not List
        """

        src_img = tf.cast(data, dtype=tf.float32)
        if src_img.shape.rank!=3 and src_img.shape.rank!=4 :
            raise ValueError("shape.rank of input data must be 3 or 4"
                            f"but got {src_img.shape.rank} @{self.__calss__.__name__}")
          
        # assert src_img.shape.rank==3 or src_img.shape.rank==4,\
        # f"shape.rank of input data must be 3 or 4, but got {src_img.shape.rank} @PreProcess_Det.__call__"
        if src_img.shape.rank==3:
            src_img = tf.expand_dims(src_img,axis=0)
        
        return self.model([src_img])   
    

#--------------------------------------------------------------------------------------------
#
#--------------------------------------------------------------------------------------------
@INFER_PROC.register_module()
class PostProcess_YOLO_Det(BaseProc):
    VESRION = '1.0.0'
    r"""PostProcess_YOLO_Det
    
    """
    '''
    YOLO_DataPrecorcess_config = {
        "Tiny": {
            "img_size": [640,640], 
            "ANCHOR_SIZE": [ [[12,16],[19,36],[40,28]], [[36,75], [76,55], [72,146]], [[142,110], [192,243],[459,401]] ],
            "STRIDES" : [8, 16, 32],
            "FEAT_SIZE": [80,40,20],
        }  
    }   
    '''

    YOLO_FPN_ANCHORS_CFG = {
            "img_size": [640,640], 
            "anchor_sizes": [ [[12,16],[19,36],[40,28]], [[36,75], [76,55], [72,146]], [[142,110], [192,243],[459,401]] ],
            "strides" : [8, 16, 32],
            "feature_map_shapes": [80,40,20],
    } 

    def __init__(self, 
            hyper_params : Optional[Dict]=None,
            batch_size : int = 1,
            num_classes  : int = 4,
            confidence_threshold :float = 0.5,
            nms_iou_threshold :float  = 0.5,
            max_detections_per_class :int = 10,
            max_detections : int = 10,
            src_img_frame : bool = True,
            single_person : bool = True, 
            name : Optional[str] = None) :
        

        '#0 verify hyper_params'
        if hyper_params is None:
            self.fpn_hyper_params = self.YOLO_FPN_ANCHORS_CFG
        else:
            if isinstance(hyper_params, dict):
                '''
                TO DO : add keys() to verify format of hyper_params
                '''
                self.fpn_hyper_params = hyper_params
            else:
                raise RuntimeError("hyper_params must be dict")    
        'YOLO_DataPrecorcess_config type'
        #self.fpn_hyper_params = self.YOLO_DataPrecorcess_config[YOLO_Model_Type]
        #fpn_shape_list =  self.fpn_hyper_params ['FEAT_SIZE'] 
        #fpn_anchor_size_list =  self.fpn_hyper_params ["ANCHOR_SIZE"]  
        #self.fpn_hyper_params = self.YOLO_FPN_ANCHORS_CFG

        fpn_shape_list =  self.fpn_hyper_params ['feature_map_shapes'] 
        fpn_anchor_size_list =  self.fpn_hyper_params ["anchor_sizes"]  
        model_input_size = self.fpn_hyper_params ["img_size"]

        # assert len(fpn_shape_list)==len(fpn_anchor_size_list),"fpn_levels in fpn_shape_list and anchor_size_list must be same"
        # assert model_input_size[0]==model_input_size[1],"ONLY SUPPORT img.shape X=Y"

        if len(fpn_shape_list)!=len(fpn_anchor_size_list):
            raise ValueError(
                            "fpn_levels in fpn_shape_list and anchor_size_list must be same"
                            f"but got fpn_shape_list: {len(fpn_shape_list)} and fpn_anchor_size_list : {(fpn_anchor_size_list)}"
            )
        
        if  model_input_size[0]!= model_input_size[1]:
            raise ValueError(
                            "ONLY SUPPORT img.shape X=Y"
                            f"but got {model_input_size}"
            )      

        self._feat_shapes_list = fpn_shape_list                    #[80,40,20]
        self._num_anchors_list = [ feat_shape*feat_shape*len(anchor_size_list) for feat_shape, anchor_size_list in zip(self._feat_shapes_list, fpn_anchor_size_list) ]                  
        self._fpn_anchor_sizes = tf.constant(fpn_anchor_size_list, dtype=tf.float32) #(3,3,2)
        
        self._grid_xy_list = []
        for feat_shape in self._feat_shapes_list: 
            grid_coords_y = tf.cast(tf.range(0, feat_shape), dtype=tf.float32)
            grid_coords_x = tf.cast(tf.range(0, feat_shape), dtype=tf.float32)
            grid_x, grid_y = tf.meshgrid(grid_coords_x, grid_coords_y)    # (grid_coords_x,grid_height)  
            grid_xy = tf.concat([grid_x[...,None],grid_y[...,None]], axis=-1) # (grid_coords_x,grid_height, 2) 
            self._grid_xy_list.append(grid_xy)

        'basisc cfg'
        self.input_size = tf.cast(model_input_size[0], dtype=tf.float32) # _input_size = (640,640)
        self.num_classes = num_classes
        self.nms_iou_threshold = nms_iou_threshold
        self.confidence_threshold = confidence_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections    
        self.src_img_frame = src_img_frame
        self.single_person_det = single_person

        super().__init__(name=name,  
                        batch_size = batch_size)
        
    def bboxes_centers2corners(self, boxes :Tensor) -> Tensor:
        '''
        cxcy,wh ->yx_min, yx_max
        [cx, cy]-[w, h]/2 = x_min, y_min
        [cx, cy]+[w, h]/2 = x_max, y_max
        '''
        boxes = tf.concat([boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],axis=-1) #xy_min,xy_max
        boxes = tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1) # yx_min, yx_max  
        return boxes    
        
    def Set_InputsTensorSpec(self, batch_size : int) -> Tuple[Tensor]: 
        """FSet_InputsTensorSpec.
        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """
        #self.inputs = [ Input(shape=TensorSpec, batch_size=self.batch_size) for TensorSpec in InputsTensorSpec ]
   
        feats = [ Input(shape=(size, size,3,self.num_classes+4+1), batch_size=batch_size) for size in self.fpn_hyper_params ['feature_map_shapes'] ]
        meta = Input(shape=(5,), batch_size=batch_size)
        InputsTensor = [feats, meta]
        return InputsTensor
    
    def forward(self,
                x : Union[List[Tensor],Tuple[Tensor]]) -> Tuple[Tensor]:
        """
        calculate Loss for per output features
        y_pred: (batch_size, 80,80,3, 4+num_cls+1)/ (batch_size, 40,40,3, 6) /(batch_size, 20,20,3, 6) from Model output
        #Note : Here, 4+1+1 means [ bbox_xywh, cls, Object_Mask]
        """
        #batch_size = y_preds_list[0].shape

        y_preds_list, meta = x


        cls_preds = []
        box_preds = []

        for fpn_level, y_pred in enumerate(y_preds_list) :

            'y_pred in one feat'
            reg_pred_i = y_pred[...,:4]            #(b,feat_h,feat_w,3,4)
            cls_pred_i = y_pred[...,4:4+self.num_classes]  #(b,feat_h,feat_w,3,mum_cls)
            obj_pred_i = y_pred[...,-1]            #(b,feat_h,feat_w,3)
            
            'decode predictions'
            grid_xy = self._grid_xy_list[fpn_level]   #(feat_h,feat_w,2) 
            feat_shapes = tf.cast( self._feat_shapes_list[fpn_level], dtype=tf.float32)  #(,) 
            anchor_size_3x2 = self._fpn_anchor_sizes[fpn_level,...] #(3,2)
            #print(reg_pred_i.shape, grid_xy.shape, feat_shapes)
            box_xy = (tf.nn.sigmoid(reg_pred_i[..., :2])*2-0.5 + grid_xy[None,:,:,None,:])/feat_shapes #(b,feat_h,feat_w,3,2)
            box_wh = ( (tf.nn.sigmoid(reg_pred_i[..., 2:4])*2)**2 )*anchor_size_3x2[None, None, None,:,:]/self.input_size #(b,feat_h,feat_w,3,2)
            box_pred_i = tf.concat([box_xy,box_wh], axis=-1) #(b,feat_h,feat_w,3,4)

            'cls prediction'
            cls_pred_i = tf.nn.sigmoid(cls_pred_i)
            obj_pred_i = tf.nn.sigmoid(obj_pred_i)
            cls_pred_i  = cls_pred_i*obj_pred_i[...,None]  #(b,feat_h,feat_w,3,mum_cls)

            'reshape'
            cls_pred_i = tf.reshape(cls_pred_i, shape=(-1, self._num_anchors_list[fpn_level], self.num_classes))
            box_pred_i = tf.reshape(box_pred_i, shape=(-1, self._num_anchors_list[fpn_level], 4))

            cls_preds.append(cls_pred_i)
            box_preds.append(box_pred_i)

        cls_preds =  tf.concat(cls_preds , axis=1)
        box_preds =  tf.concat(box_preds , axis=1)
        box_preds = tf.expand_dims(box_preds, axis=2)          #(b,25200,1,4)
        box_preds = self.bboxes_centers2corners(box_preds)
        box_preds = tf.tile(box_preds,[1,1,4,1])*self.input_size     #(b,25200,4,4)  

        
        'Decode PD predictions '
        # https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
        decoded_detections = tf.image.combined_non_max_suppression(
                                box_preds,
                                cls_preds,
                                self.max_detections_per_class,
                                self.max_detections,
                                self.nms_iou_threshold,
                                self.confidence_threshold,
                                clip_boxes=False,
                            )     
        
        if self.src_img_frame :
            'coordinator transform to src img frame'
            aspect_resize2src = meta[:,0] #(1,)
            offset_yx = meta[:,1:3] #(1,2)
            src_img_shape_yx = meta[:,3:5] #(1,2)

            rt = (decoded_detections.nmsed_boxes[:,:,:2]-offset_yx[:,None,:])/aspect_resize2src #(1, max_detections,2) 
            lb = (decoded_detections.nmsed_boxes[:,:,2:4]-offset_yx[:,None,:])/aspect_resize2src #(1, max_detections,2) 
            rt = tf.maximum(rt, tf.cast( [0.,0.], dtype=tf.float32))
            #lb = tf.minimum(lb, tf.cast( [self.input_size ,self.input_size ]/aspect_resize2src, dtype=tf.float32))
            lb = tf.minimum(lb,src_img_shape_yx[:,None,:])

            nmsed_src_boxes = tf.concat([rt,lb], axis=-1) ##(1, max_detections,4)
        else:
            nmsed_src_boxes = decoded_detections.nmsed_boxes    

        'topk=10 bbox type = y1,x1,y2,x2 (yx_min, yx_max)'
        if self.single_person_det :
            'obtain the best bboxe for single person if it is availabe @max_detections'
            mask = tf.equal(decoded_detections.nmsed_classes, 1.) #(b, max_detections)
            detection_scores_person = tf.where(mask, decoded_detections.nmsed_scores, 0.) #(b,max_detections) 

            cond = tf.reduce_any(mask, axis=-1) #(b,)
            indices = tf.expand_dims( tf.argmax(detection_scores_person, axis=-1) , axis=-1)  #(b,) =>(b,1)
            x = tf.gather(nmsed_src_boxes, indices=indices,  batch_dims=-1, axis=1) #(b,1,4)
            if self.src_img_frame: 
                null_bbox = [0.,0.,meta[0,3],meta[0,4]]#(1,2), self.input_size]
            else :
                null_bbox = [0.,0.,self.input_size, self.input_size]

            personal_bbox_yx = tf.where(cond[:,None],
                                        tf.squeeze(x,axis=1),
                                        null_bbox
                                ) #(b,4) 
        else:
            'need to verify'
            mask = tf.equal(decoded_detections.nmsed_classes, 1.) #(b, max_detections)
            x = tf.boolen_mask(nmsed_src_boxes,mask)  #(b, n, 4)
            '''
            personal_bbox_yx = tf.where(tf.expand_dims(mask, axis=-1),
                                        nmsed_src_boxes,
                                        [0.,0.,0.,0.]
                                        ) #(b,4)
            '''

        #return personal_bbox_yx, decoded_detections  
        personal_bbox_yx = tf.keras.layers.Layer(name="pred_bbox_out")(personal_bbox_yx)
        return personal_bbox_yx, decoded_detections
    
    @tf.function(reduce_retracing=False)
    def __call__(self, 
                feats_list: List[Tensor],
                meta : Tensor) ->Tuple[Tensor]:
        

        # if len(feats_list)!=3:
        #     raise ValueError("output feats num of yolo model must be 3"
        #                     f"but got {len(feats_list)} @{self.__calss__.__name__}") 
        
        # if feats_list[0].shape.rank!=5:
        #    raise ValueError("shape.rank of feats must be 5"
        #                    f"but got {feats_list[0].shape.rank} @{self.__class__.__name} -- call")
        
        # if meta.shape!=(1,5):
        #    raise ValueError("meta.shape must be (1,5)"
        #                    f"but got {meta.shape} @{self.__class__.__name} -- call")
        

        # assert feats_list[0].shape.rank==5 and len(feats_list)==3, \
        # f"shape.rank of feats must be 5 @{self.__class__.__name} -- call"

        # assert meta.shape.rank==2, \
        # f"shape.rank of meta must be 2 @{self.__class__.__name} -- call"

        return self.model([feats_list, meta])   
 
