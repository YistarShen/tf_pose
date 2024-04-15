from typing import Dict, List, Optional, Tuple, Union, Any
from tensorflow import Tensor
import tensorflow as tf
from lib.Registers import CODECS
from .base_codec import BaseCodec
from lib.datasets.transforms import BBoxesFormatTransform
from lib.datasets.transforms.utils import PackInputTensorTypeSpec
class PAFPN_AnchorBox:
    def __init__(self, fpn_params, anchor_normalize=False):
        self.scale_normalize = anchor_normalize

        'Image Size'
        self.image_height = tf.cast(fpn_params['img_size'][0], dtype=tf.float32)
        self.image_width = tf.cast(fpn_params['img_size'][1], dtype=tf.float32)
        self.img_shape_xy = tf.stack([self.image_width, self.image_height],axis=0)
        'anchor size for feats'
        self.fpn_anchor_sizes = tf.constant(fpn_params["anchor_sizes"], dtype=tf.float32) #(3,3,2)
        self.anchor_num_per_pixel = len(fpn_params['anchor_sizes'][0]) # 3
        self.strides = fpn_params["strides"]
        self.fpn_levels = len(self.strides)

    def show_info(self,  fpn_anchors_list, fpn_anchors):
        print(f"input image_shape :{self.image_height}x{self.image_width} ---(anchor_normalize :{self.scale_normalize})")
        for i in range(len(fpn_anchors_list)):
            print(f"fpn_anchors_level_{i}: {fpn_anchors_list[i].shape} @strids={self.strides[i]}")
            
        print(f"fpn_anchors_shape :{fpn_anchors.shape}")

    def center_anchors_generator(self, anchor_num_per_pixel, grid_height, grid_width):

        """Generating top left anchors for given anchor_ratios, anchor_scales and image size values.
        inputs:
            hyper_params = dictionary
        outputs:
            anchor_centers = (anchor_counts, [y1, x1])  @shape=(grid_coords_x,grid_height,2)
        """

        stride_y = 1 / grid_height
        stride_x = 1 / grid_width
        grid_coords_y = tf.cast(tf.range(0, grid_height) / grid_height + stride_y / 2, dtype=tf.float32) #(grid_h,)
        grid_coords_x = tf.cast(tf.range(0, grid_width) / grid_width + stride_x / 2, dtype=tf.float32)  #(grid_w,)
        grid_x, grid_y = tf.meshgrid(grid_coords_x, grid_coords_y)  # (grid_h,grid_w) and (grid_h,grid_w)
        grid_x = tf.expand_dims(grid_x, axis=-1)            # (grid_h,grid_w,1)
        grid_y = tf.expand_dims(grid_y, axis=-1)            # (grid_h,grid_w,1)
        anchor_centers_xy = tf.concat([grid_x, grid_y], axis=-1)   # (grid_h,grid_w,2)

        if self.scale_normalize==False : 
            anchor_centers_xy = anchor_centers_xy*self.img_shape_xy # (grid_h,grid_w,2)

        anchor_centers_xy = tf.tile(anchor_centers_xy[:,:,None,:], [1, 1, anchor_num_per_pixel, 1])  #(grid_h,grid_w,2)=>(grid_h,grid_w,3,2)

        '''
        anchor_centers_xy : (grid_h,grid_w,3,2)
        type of anchor_centers_xy is center_xy : (x,y)   
        ''' 
        return anchor_centers_xy

    def base_anchors_generator(self, fpn_anchor_sizes, grid_height, grid_width):
        """Generating top left anchors for given anchor_ratios, anchor_scales and image size values.
        inputs:
            fpn_anchor_sizes : (3,2)
        outputs:
            anchor_centers = (anchor_counts, [y1, x1])  @shape=(grid_coords_x,grid_height,2
        """
        anchors_bases_wh = fpn_anchor_sizes                                #(3,2), wh
        anchors_bases_wh = tf.tile(anchors_bases_wh[None,None,:,:],(grid_height, grid_width, 1, 1)) #(grid_h,grid_w,3,2)
        if self.scale_normalize :
            anchors_bases_wh = anchors_bases_wh/self.img_shape_xy                  #(grid_h,grid_w,3,2)  
        return anchors_bases_wh 

    def __call__(self, show_info=True):

        fpn_anchors_list = [] # target is [ [80,80,3,4], [40,40,3,4], [20,20,3,(x,y,w,h)] ]
        feats_h = tf.cast(tf.math.ceil(self.image_height/self.strides),dtype=tf.int32)  #tf.Tensor([80 40 20], shape=(3,), dtype=int32)
        feats_w = tf.cast(tf.math.ceil(self.image_width/self.strides),dtype=tf.int32)


        for level in range(self.fpn_levels):
            feat_h = feats_h[level]
            feat_w = feats_w[level]
            anchor_sizes = self.fpn_anchor_sizes[level,:,:] #(3,2)
    
            anchor_centers_xy = self.center_anchors_generator(self.anchor_num_per_pixel, 
                                    grid_height = feat_h, 
                                    grid_width = feat_w)     #(feat_h,feat_w,3,2)


            anchors_bases_wh = self.base_anchors_generator(anchor_sizes,
                                    grid_height = feat_h, 
                                    grid_width = feat_w)      #(feat_h,feat_w,3,2)

            'combine anchor_centers and bases'
            anchors_level_xywh = tf.concat([anchor_centers_xy, anchors_bases_wh], axis=-1) #(feat_h,feat_w,3,4) 
            fpn_anchors_list.append(anchors_level_xywh)                      #[ (feat_h,feat_w,3,4),...]

        fpn_anchors = []
        for fpn_anchors_level in fpn_anchors_list:
            fpn_anchors_level = tf.reshape(fpn_anchors_level,(-1,4))       #(feat_h*feat_w*3,4)  
            fpn_anchors.append(fpn_anchors_level)  

        fpn_anchors = tf.concat(fpn_anchors, axis=0)  

        if (show_info == True):
            self.show_info(fpn_anchors_list, fpn_anchors)
    
        return fpn_anchors_list, fpn_anchors
    

@CODECS.register_module()
class  YoloAnchorBaseCodec(BaseCodec):
    VERSION = '1.1.0'
    ENCODER_USE_PRED = False
    ENCODER_GEN_SAMPLE_WEIGHT = False

    YOLO_FPN_ANCHORS_CFG = {
            "img_size": [640,640], 
            "anchor_sizes": [ [[12,16],[19,36],[40,28]], [[36,75], [76,55], [72,146]], [[142,110], [192,243],[459,401]] ],
            "strides" : [8, 16, 32],
            "feature_map_shapes": [80,40,20],
            "fpn_balance" : [4., 1., 0.4],
    }

    def __init__(self, 
            hyper_params : Optional[dict]=None,
            num_classes : int = 4,
            label_start_id : int = 1,
            gt_bbox_format :str = 'xywh', 
            confidence_threshold :float = 0.5,
            nms_iou_threshold :float  = 0.5,
            max_detections_per_class :int = 10,
            max_detections : int = 10,
            wrapped_y_true=True,
            **kwargs):
        
        super().__init__(**kwargs)
        
        """ Tags the result of function by setting _is_zeros_tensor attribute.
        Input kps : (x, y, vis)
        Output heatmap_size : (w,h) or (x,y)
        Simple Coordinate Classification (simcc)

        """
        '#0 verify hyper_params'
        if hyper_params is None:
            self.hyper_params = self.YOLO_FPN_ANCHORS_CFG
        else:
            if isinstance(hyper_params, dict):
                '''
                TO DO : add keys() to verify format of hyper_params
                '''
                self.hyper_params = hyper_params
            else:
                raise RuntimeError("hyper_params must be dict")    
            
        '#1 encode part'
        fpn_anchors_gen = PAFPN_AnchorBox(fpn_params=self.hyper_params, anchor_normalize=True)
        norm_fpn_anchors_list, _ = fpn_anchors_gen(show_info=True) #[(feat_h,feat_w,3,4), .... ]
        self.norm_fpn_anchors_list = [ tf.cast(fpn_anchors,dtype=self.compute_dtype) for fpn_anchors in norm_fpn_anchors_list]

        self.img_shape_yx = tf.constant(self.hyper_params['img_size'], dtype=self.compute_dtype) #(2,)
        self.img_shape_xy = tf.reverse(self.img_shape_yx, axis=[0]) #(2,)

        
        '#2 decode part'
        self.num_classes = num_classes

        fpn_shape_list =  self.hyper_params['feature_map_shapes'] 
        fpn_anchor_size_list =  self.hyper_params["anchor_sizes"]  

        assert len(fpn_shape_list)==len(fpn_anchor_size_list),"fpn_levels in fpn_shape_list and anchor_size_list must be same"
        assert self.img_shape_xy [0]==self.img_shape_xy [1],"ONLY SUPPORT img.shape X=Y (model input)"

        self._feat_shapes_list = fpn_shape_list                    #[80,40,20]
        self._num_anchors_list = [ feat_shape*feat_shape*len(anchor_size_list) for feat_shape, anchor_size_list in zip(self._feat_shapes_list, fpn_anchor_size_list) ]                  
        self._fpn_anchor_sizes = tf.constant(fpn_anchor_size_list, dtype=self.compute_dtype) #(3,3,2)
        
        self._grid_xy_list = []
        for feat_shape in self._feat_shapes_list: 
            grid_coords_y = tf.cast(tf.range(0, feat_shape), dtype=self.compute_dtype)
            grid_coords_x = tf.cast(tf.range(0, feat_shape), dtype=self.compute_dtype)
            grid_x, grid_y = tf.meshgrid(grid_coords_x, grid_coords_y)    # (grid_coords_x,grid_height)  
            grid_xy = tf.concat([grid_x[...,None],grid_y[...,None]], axis=-1) # (grid_coords_x,grid_height, 2) 
            self._grid_xy_list.append(grid_xy)

        self.input_size = tf.cast(self.img_shape_xy[0] , dtype=self.compute_dtype) # _input_size = (640,640)
        self.num_classes = num_classes
        self.nms_iou_threshold = nms_iou_threshold
        self.confidence_threshold = confidence_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections    
    
        self.label_id_shitft = -label_start_id
        if gt_bbox_format != 'cxcywh':
            self.gt_bboxes_format_transform = BBoxesFormatTransform(
                convert_type=gt_bbox_format+'2cxcywh'
            )
            print(f"apply bboxes_format_transform <{gt_bbox_format+'2cxcywh'}> ")

    def Get_ANCHORS_CFG(self) -> Dict:
        return self.hyper_params
    
    def _gen_yolo_fpn_tragets(self, norm_fpn_anchors, norm_gt_bboxes, cls_ids):
        #---------------------------------------------------------------
        # norm_fpn_anchors : (feat_h, feat_w, 3, 4),  @box_type=ctr_xywh
        # norm_gt_bboxes : (b, num_gt_bbox, 4),     @box_type=ctr_xywh
        # cls_ids : (b, num_gt_bbox)
        # All anchors and bboxies were normalized by img_shape_xy
        #-----------------------------------------------------------------
        feat_shape = norm_fpn_anchors.shape[0] #feat_h=feat_w
        norm_bboxes_xy = norm_gt_bboxes[...,:2]   #(b,num_gt_bbox,2)
        norm_bboxes_wh = norm_gt_bboxes[...,2:4]   #(b,num_gt_bbox,2)

        
        '#1 mask for padding sample'
        bboxes_prod_wh = tf.math.reduce_prod(norm_bboxes_wh, axis=-1)  #(b,num_gt_bbox,2) =>(b,num_gt_bbox)
        eff_mask = tf.greater(bboxes_prod_wh, 0.)          #(b,num_gt_bbox) @bool

        '#2 ctr mask'
        gt_bboxes_xy_feat = norm_bboxes_xy*feat_shape                              #(b,num_gt_boxes,2)@feat frame, float
        fpn_anchors_xy_feat = norm_fpn_anchors[...,:2]*feat_shape                       #(feat_h,feat_w,3,2)@feat frame, float
        delta_ctr_xy = fpn_anchors_xy_feat[None,:,:,:,None,:] - gt_bboxes_xy_feat[:,None, None, None,:,:]  #(b,feat_h,feat_w,3,num_gt_boxes,2)
        abs_delta_ctr_xy = tf.abs(delta_ctr_xy)                                   #(b,feat_h,feat_w,3,num_gt_boxes,2)

        cond_ctr = tf.less_equal(abs_delta_ctr_xy, 0.5)                                    #(b,feat_h,feat_w,3,num_gt_boxes,2)
        cond_near = tf.less_equal(abs_delta_ctr_xy, 1.)                                    #(b,feat_h,feat_w,3,num_gt_boxes,2)
        cond_near_left_and_right = tf.stack([cond_near[...,0], cond_ctr[...,1] ], axis=-1) #(b,feat_h,feat_w,3,num_gt_boxes,2)
        cond_near_top_and_down = tf.stack([ cond_ctr[...,0], cond_near[...,1]], axis=-1)   #(b,feat_h,feat_w,3,num_gt_boxes,2)

        cond_ctr = tf.math.reduce_all(cond_ctr, axis=-1, keepdims=False)                                 #(b,feat_h,feat_w,3,num_gt_boxes,2) => (b,feat_h,feat_w,3,num_gt_boxes)
        cond_near_left_and_right = tf.math.reduce_all(cond_near_left_and_right, axis=-1, keepdims=False) #(b,feat_h,feat_w,3,num_gt_boxes,2) => (b,feat_h,feat_w,3,num_gt_boxes)
        cond_near_top_and_down = tf.math.reduce_all(cond_near_top_and_down, axis=-1, keepdims=False)     #(b,feat_h,feat_w,3,num_gt_boxes,2) => (b,feat_h,feat_w,3,num_gt_boxes)

        ctr_mask = tf.stack([cond_ctr, cond_near_left_and_right, cond_near_top_and_down], axis=-1) #(b,feat_h,feat_w,3,num_gt_boxes)  => (b,feat_h,feat_w,3,num_gt_boxes,3)
        ctr_mask = tf.reduce_any(ctr_mask, axis=-1)                                                #(b,feat_h,feat_w,3,num_gt_boxes,3) => (b,feat_h,feat_w,3,num_gt_boxes)                                           
        eff_mask = tf.math.logical_and(ctr_mask, eff_mask[:,None, None, None,:])                   #(b,feat_h,feat_w,3,num_gt_boxes)***

        '#3 wh'
        ratios_of_gt_anchors = norm_bboxes_wh[:,None,:,:] / norm_fpn_anchors[None, 0, 0, :, None, 2:4]   #(b,3,num_gt_boxes,2)
        ratios_of_anchors_gt = norm_fpn_anchors[None, 0, 0, :, None, 2:4] / norm_bboxes_wh[:,None,:,:]   #(b,3,num_gt_boxes,2)
        ratios = tf.concat([ratios_of_gt_anchors,ratios_of_anchors_gt], axis=-1)                 #(b,3,num_gt_boxes,4)
        max_ratios = tf.reduce_max(ratios, axis=-1)                                 #(b,3,num_gt_boxes)

        '#4 combine generated targets'
        yolo_conf = tf.where(eff_mask, max_ratios[:,None, None, :,:], 4.) #(b,feat_h,feat_w,3,num_gt_boxes)
        eff_mask = tf.less(tf.reduce_min(yolo_conf, axis=-1), 4.)      #(b,feat_h,feat_w,3,num_gt_boxes) => (b,feat_h,feat_w,3)

        '4-1 assign class' 
        matched_gt_idx = tf.math.argmin(yolo_conf, axis=-1)             #(b,feat_h,feat_w,3,num_gt_boxes) => (b,feat_h,feat_w,3)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx, batch_dims=1)   #(b,num_gt_boxes) => (b,feat_h,feat_w,3)
        cls_targets = tf.where(eff_mask, matched_gt_cls_ids, 0)          #(b,feat_h,feat_w,3)

        '4-2 assign bbox'
        matched_gt_boxes = tf.gather(norm_gt_bboxes, matched_gt_idx, batch_dims=1)       #(b,num_gt_boxes,4) => (b,feat_h,feat_w,3,4)
        matched_gt_boxes = tf.where(eff_mask[...,None], matched_gt_boxes, 0.)          #(b,feat_h,feat_w,3,4)

        '4-3 assign object confidence'
        eff_mask = tf.cast(eff_mask, dtype=self.compute_dtype)   #(b,feat_h,feat_w,3) positive samples

        '4-4 fpn_tragets'
        feat_targets = tf.concat([matched_gt_boxes, cls_targets[...,None], eff_mask[...,None]], axis=-1) #(b,feat_h,feat_w,3,4+1+1)

        return feat_targets
    def encode(self, 
            data : Dict[str,tf.Tensor]) ->Tuple[tf.Tensor, List[tf.Tensor]]: 
            
        return self.batch_encode(data)
    
    
    def gen_targets(self, 
                    gt_boxes_ctr_xywh : Tensor, 
                    gt_labels : Tensor):

        gt_labels = tf.cast(gt_labels, dtype=self.compute_dtype)

        norm_gt_bboxes = gt_boxes_ctr_xywh/tf.tile(self.img_shape_xy,(2,))    
        #(b,num_gt_boxes,4)
        feat_targets_list = []
        for norm_fpn_anchors in self.norm_fpn_anchors_list:
            feat_targets = self._gen_yolo_fpn_tragets(
                norm_fpn_anchors, 
                norm_gt_bboxes, 
                gt_labels
            )
            if  self.embedded_codec :
                feat_targets = tf.stop_gradient(feat_targets)
            
            feat_targets_list.append(feat_targets) 

        return feat_targets_list
            

    
    def batch_encode(self, 
                    data : Dict[str,tf.Tensor],
                    y_pred  = None)-> Dict[str,tf.Tensor]:
        
        gt_bboxes =tf.cast( 
            data['bbox'] , dtype=self.compute_dtype
        )
        if hasattr(self,'gt_bboxes_format_transform'):
            gt_bboxes = self.gt_bboxes_format_transform(gt_bboxes) #(b,num_gt_bbox,4) xywh=>xyxy  @image_frames
       
        #cls_ids = data['labels'] 
        gt_labels = data["labels"] + self.label_id_shitft #cls_ids


        feat_targets_list = self.gen_targets(
            gt_bboxes, gt_labels
        )

        # '#0 init'
        # cls_ids = tf.cast(cls_ids, dtype=self.compute_dtype)
        # norm_gt_bboxes = gt_boxes_ctr_xywh/tf.tile(self.img_shape_xy,(2,))           #(b,num_gt_boxes,4)
        # feat_targets_list = []
        # for norm_fpn_anchors in self.norm_fpn_anchors_list:
        #     feat_targets = self._gen_yolo_fpn_tragets(norm_fpn_anchors, norm_gt_bboxes, cls_ids)
        #     feat_targets_list.append(feat_targets)                                   # [(b,feat_h,feat_w,3,6),.... ]

        'invalid type of data value in dict must be tuple[tf.Tensor] or tf.Tensor not list'    
        data['y_true'] = (*feat_targets_list,)
        return data
    
    def decode(self, 
               y_preds_list : tf.Tensor) -> tf.Tensor:
        return self.batch_decode(y_preds_list)
    
    def batch_decode(
            self, 
            y_preds_list : List[tf.Tensor], 
            meta_data : Optional[dict] = None,
            *args, **kwargs
        ) -> object:
        
        """
        calculate Loss for per output features
        y_pred: (batch_size, 80,80,3, 4+num_cls+1)/ (batch_size, 40,40,3, 6) /(batch_size, 20,20,3, 6) from Model output
        #Note : Here, 4+1+1 means [ bbox_xywh, cls, Object_Mask]
        """
        cls_preds = []
        box_preds = []
        for fpn_level, y_pred in enumerate(y_preds_list) :

            'y_pred in one feat'
            reg_pred_i = y_pred[...,:4]            #(b,feat_h,feat_w,3,4)
            cls_pred_i = y_pred[...,4:4+self.num_classes]  #(b,feat_h,feat_w,3,mum_cls)
            obj_pred_i = y_pred[...,-1]            #(b,feat_h,feat_w,3)
            
            'decode predictions'
            grid_xy = self._grid_xy_list[fpn_level]   #(feat_h,feat_w,2) 
            feat_shapes = tf.cast( self._feat_shapes_list[fpn_level], dtype=self.compute_dtype)  #(,) 
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
            cls_pred_i = tf.reshape(cls_pred_i, shape=(-1, self._num_anchors_list[fpn_level], self.num_classes))   #(b,feat_h*feat_w*3,mum_cls)
            box_pred_i = tf.reshape(box_pred_i, shape=(-1, self._num_anchors_list[fpn_level], 4))

            cls_preds.append(cls_pred_i)
            box_preds.append(box_pred_i)

        cls_preds =  tf.concat(cls_preds , axis=1)            #(b,25200,4)
        box_preds =  tf.concat(box_preds , axis=1)
        box_preds = tf.expand_dims(box_preds, axis=2)          #(b,25200,1,4)
        box_preds = self.bboxes_centers2corners(box_preds)
        box_preds = tf.tile(box_preds,[1,1,4,1])*self.input_size     #(b,25200,4,4)  

        
        'Decode bbox and cls predictions by tf.image.combined_non_max_suppression'
        # https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
        return tf.image.combined_non_max_suppression(
                                box_preds,
                                cls_preds,
                                self.max_detections_per_class,
                                self.max_detections,
                                self.nms_iou_threshold,
                                self.confidence_threshold,
                                clip_boxes=False,
                            )    
     
    def bboxes_centers2corners(self,
                            boxes :Tensor) -> Tensor:
        '''
        cxcy,wh ->yx_min, yx_max
        [cx, cy]-[w, h]/2 = x_min, y_min
        [cx, cy]+[w, h]/2 = x_max, y_max
        '''
        boxes = tf.concat([boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],axis=-1) #xy_min,xy_max
        boxes = tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1) # yx_min, yx_max  
        return boxes  
    
    def vis_encoder_res(
            self, 
            tfrec_datasets_list : Union[List, Dict], 
            transforms_list  : Optional[Union[List, Dict]]=None, 
            take_batch_num : int = 1,
            vis_fn : Optional[callable] =  None,
            batch_ids :  Union[List[int], int] = [0],  
            show_anchors  : bool = False
        ):
            
            if vis_fn is None :
               vis_fn =  Vis_AnchorBaseDetSampleCodec
    
            if transforms_list is None :
                from lib.datasets.transforms import RandomPadImageResize
                img_resize = RandomPadImageResize(
                    target_size = self.hyper_params['img_size'],
                    pad_types  = ['center'],
                    pad_val = 0,
                    use_udp  = False,
                )
                transforms_list = [img_resize]
            
            batch_dataset = super().gen_tfds_w_codec(
                tfrec_datasets_list = tfrec_datasets_list, 
                transforms_list = transforms_list, 
                test_mode=True, 
                batch_size=16, 
                shuffle=True
            )
            batched_samples = [ feats for feats in batch_dataset.take(take_batch_num)]

            vis_fn(
                batched_samples,    
                batch_ids,
                anchor_size_list = self.hyper_params['anchor_sizes'],
                show_anchors = show_anchors
            )
            del batch_dataset


def Vis_AnchorBaseDetSampleCodec( 
        batched_samples: Union[Dict, List],    
        batch_ids :  Union[List[int], int] = [0],
        anchor_size_list : Optional[list] = None,
        show_anchors : bool = False
):  
    import matplotlib.pyplot as plt 
    def plot(
            image, 
            fpn_targets_list, 
            anchor_size_list, 
            show_anchors,
            gt_bbox,
            gt_labels
    ):
        r'''
        image : (h,w,3) 
        fpn_targets_list : [ (p4_h,45_w,3,6), (p4_h,p4_w,3,6), (p5_h,p5_w,3,6)]
        '''
        fpn_level_colors = ['white','silver','yellow']
        anchor_size_level_colors = ['r','g','b']

        title = 'Yolo_AnchorBaseEncoder'
        sub_plots = len(fpn_targets_list)+1


        img_shape_yx = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        img_shape_yx = tf.shape(image)[:2]
        plt.figure(figsize=(30,10))
        plt.subplot(1,sub_plots,1)
        plt.title( "src_image", fontsize = 12)
        plt.imshow(image.numpy())
        for bbox, label in zip(gt_bbox,gt_labels):
            if tf.reduce_prod(bbox[2:])<=0.:
                continue
            ax = plt.gca()
            x1,y1,w,h = bbox
            patch = plt.Rectangle(
                [x1, y1], w, h, 
                fill=False, 
                edgecolor='yellow', 
                linewidth=2
            )
            ax.add_patch(patch)
            ax.text(
                x1,y1,
                "{}".format(int(label)), 
                size=7,
                bbox={"facecolor": [1, 0, 1], "alpha": 0.8},
                clip_box=ax.clipbox,
                clip_on=True,
            )

        for fpn_level, feat_targets_3x6 in enumerate(fpn_targets_list):

            if show_anchors and isinstance(anchor_size_list,list):
                anchor_wh_list = anchor_size_list[fpn_level]

            feat_shape = tf.shape(feat_targets_3x6)[:2]
            plt.subplot(1,sub_plots,fpn_level+2)
            plt.title(
                title+f"\n  fpn level-{fpn_level+1} : {feat_shape[0]}x{feat_shape[1]}", fontsize = 12
            )
            plt.imshow(image.numpy())
            
            'anchor size level'
            for anchor_size_level in range( feat_targets_3x6.shape[2] ):
                feat_targets = feat_targets_3x6[:,:,anchor_size_level,:]
                feat_bboxes = feat_targets[:,:,:4] #(feat_h,feat_w,4)
                feat_obj = feat_targets[:,:,-1]
                feat_cls = feat_targets[:,:,-2] #(feat_h,feat_w) , eff_mask 
                #src_cls_indices_yx = tf.where(feat_cls-label_id_shitft)*(2**(3+fpn_level))
                mask = tf.equal(feat_obj,1.)
                src_cls_indices_yx = tf.where(mask)*(2**(3+fpn_level))
                bboxes = tf.boolean_mask(feat_bboxes, mask, axis=0)
                bboxes *= tf.tile( tf.cast(img_shape_yx, dtype=bboxes.dtype),(2,))
                classes = tf.boolean_mask(feat_cls, mask, axis=0)
                if False :
                    print(
                        f'eff_bboxes: {bboxes.shape}  ( fpn_id-{fpn_level} @{feat_targets_3x6.shape[:2]}, anchor_size_id-{anchor_size_level} )'
                    )
                for bbox, cls_id, src_cls_index_yx in zip(bboxes,classes,src_cls_indices_yx):
                    ax = plt.gca()
                    'bboxes_cxcy_wh'
                    cx, cy, w, h = bbox
                    x1, x2 = cx-w/2.0, cx+w/2.0
                    y1, y2 = cy-h/2.0, cy+h/2.0

                    patch = plt.Rectangle(
                        [x1, y1], w, h, 
                        fill=False, 
                        edgecolor='yellow', 
                        linewidth=2
                    )
                    ax.add_patch(patch)

                    text = "{}".format(int(cls_id))
                    #text = f"{int(cls_id)} --<{fpn_level}-{anchor_size_level}>"
                    #text2 = f"<{fpn_level}-{anchor_size_level}>"
                    color = [1, 0, 1]
                    ax.text(
                            x1,y1,
                            text, 
                            size=7,
                            bbox={"facecolor": 'yellow', "alpha": 0.8},
                            clip_box=ax.clipbox,
                            clip_on=True,
                    )
                    if 'anchor_wh_list' in locals() : 
                        anchor_wh = anchor_wh_list[anchor_size_level]
                        anchor_yx = src_cls_index_yx -tf.cast([anchor_wh[1],anchor_wh[0]], dtype=src_cls_index_yx.dtype)//2
                        anchor_patch = plt.Rectangle(
                            [anchor_yx[1], anchor_yx[0]], 
                            anchor_wh[0], anchor_wh[1], 
                            fill=False, 
                            edgecolor=anchor_size_level_colors[anchor_size_level], 
                            linewidth=1
                        )
                        ax.add_patch(anchor_patch)
                        ax.text(
                                anchor_yx[1],anchor_yx[0],
                                text, 
                                size=5,
                                bbox={"facecolor": color, "alpha": 0.8},
                                clip_box=ax.clipbox,
                                clip_on=True,
                        )                  
                    circle_outter = plt.Circle((src_cls_index_yx[1], src_cls_index_yx[0]), 3, color=fpn_level_colors[fpn_level])
                    circle_inner = plt.Circle((src_cls_index_yx[1], src_cls_index_yx[0]), 1, color=anchor_size_level_colors[anchor_size_level])
                    ax.add_patch(circle_outter)
                    ax.add_patch(circle_inner)

    PackInputTensorTypeSpec(batched_samples[0],{}, show_log=True)
    print("\n\n\n")
    for features in batched_samples:
        for batch_id in batch_ids :
            fpn_targets_list = [ ith_fpn_feat[batch_id] for ith_fpn_feat in features['y_true']]
            image = features['image'][batch_id]
            plot( 
                image, 
                fpn_targets_list, 
                anchor_size_list,
                show_anchors,
                features['bbox'][batch_id],
                features['labels'][batch_id],
            )




    # def vis_encoder_res(
    #         self, 
    #         batched_samples : Union[Dict, List], 
    #         batch_ids :  Union[List[int], int] = [0],  
    #         figsize : Tuple[int] = (10,10)
    #     ):
    #     import matplotlib.pyplot as plt
    #     fpn_level_colors = ['white','silver','yellow']
    #     anchor_size_level_colors = ['r','g','b']

    #     if not isinstance(batched_samples, (dict, list)):
    #         raise TypeError(
    #             "input samples must be dict or List"
    #         )
    #     if isinstance(batched_samples, dict):
    #         batched_samples = [batched_samples]
    #     if isinstance(batch_ids, int):
    #         batch_ids = [batch_ids]

       
    #     def plot(feats_targets_list, label_id_shitft):

    #         for fpn_level, feat_targets_3x6 in enumerate(feats_targets_list):
    #             feat_shape = tf.shape(feat_targets_3x6)[:2]
    #             'anchor size level'
    #             for anchor_size_level in range( feat_targets_3x6.shape[2] ):
    #                 feat_targets = feat_targets_3x6[:,:,anchor_size_level,:]
    #                 feat_bboxes = feat_targets[:,:,:4] #(feat_h,feat_w,4)
    #                 feat_obj = feat_targets[:,:,-1]
    #                 feat_cls = feat_targets[:,:,-2] #(feat_h,feat_w)
    #                 src_cls_indices_yx = tf.where(feat_cls-label_id_shitft)*(2**(3+fpn_level))
    #                 mask = tf.equal(feat_obj,1.)
    #                 bboxes = tf.boolean_mask(feat_bboxes, mask, axis=0)*tf.tile( img_shape_yx,(2,))
    #                 classes = tf.boolean_mask(feat_cls, mask, axis=0)
    #                 print(f'eff_bboxes: {bboxes.shape}  ( fpn_id-{fpn_level} @{feat_targets_3x6.shape[:2]}, anchor_size_id-{anchor_size_level} )')
    #                 for bbox, cls_id, src_cls_index_yx in zip(bboxes,classes,src_cls_indices_yx):
    #                     ax = plt.gca()
    #                     'bboxes_cxcy_wh'
    #                     cx, cy, w, h = bbox
    #                     x1, x2 = cx-w/2.0, cx+w/2.0
    #                     y1, y2 = cy-h/2.0, cy+h/2.0

    #                     patch = plt.Rectangle([x1, y1], w, h, fill=False, edgecolor=[0, 0, 1], linewidth=2)
    #                     ax.add_patch(patch)
    #                     text = "{}".format(int(cls_id))
    #                     #text = f"{int(cls_id)} --<{fpn_level}-{anchor_size_level}>"
    #                     text2 = f"<{fpn_level}-{anchor_size_level}>"
    #                     color = [1, 0, 1]
    #                     ax.text(
    #                         x1,y1,text, size=7,
    #                         bbox={"facecolor": color, "alpha": 0.8},
    #                         clip_box=ax.clipbox,
    #                         clip_on=True,
    #                     )

    #                     circle_outter = plt.Circle((src_cls_index_yx[1], src_cls_index_yx[0]), 3, color=fpn_level_colors[fpn_level])
    #                     circle_inner = plt.Circle((src_cls_index_yx[1], src_cls_index_yx[0]), 1, color=anchor_size_level_colors[anchor_size_level])
    #                     ax.add_patch(circle_outter)
    #                     ax.add_patch(circle_inner)



    #     # batch_feats_targets_list = batched_samples['y_true']
    #     # batch_images = batched_samples['image']
    #     # print(f"Image shape: {batch_images.shape}")
    #     # print(f"feat_num: {len(batch_feats_targets_list)}")
    #     # for fpn_level, batch_feat_targets in enumerate(batched_samples[0]['y_true']):
    #     #     print(f"feat_targets-{fpn_level+1} shape: {batch_feat_targets.shape}")
    #     # img_shape_yx = tf.cast( tf.shape(batch_images)[1:3], dtype=self.compute_dtype)

    #     for features in enumerate(batched_samples) :
    #         for batch_id in batch_ids :
    #             feats_targets_list = [ ith_fpn_feat[batch_id] for ith_fpn_feat in features['y_true']]
    #             image = features['image'][batch_id].numpy()



    #             plt.figure(figsize=figsize)
    #             plt.imshow(image)
    #             plot(feats_targets_list, self.label_id_shitft)
            



# @CODECS.register_module()
# class  YoloAnchorBase(BaseBBoxesCodec):

#     YOLO_FPN_ANCHORS_CFG = {
#             "img_size": [640,640], 
#             "anchor_sizes": [ [[12,16],[19,36],[40,28]], [[36,75], [76,55], [72,146]], [[142,110], [192,243],[459,401]] ],
#             "strides" : [8, 16, 32],
#             "feature_map_shapes": [80,40,20],
#             "fpn_balance" : [4., 1., 0.4],
#     }

#     def __init__(self, 
#             hyper_params : Optional[dict]=None,
#             num_classes : int = 4,
#             confidence_threshold :float = 0.5,
#             nms_iou_threshold :float  = 0.5,
#             max_detections_per_class :int = 10,
#             max_detections : int = 10,
#             wrapped_y_true=True):
        
#         """ Tags the result of function by setting _is_zeros_tensor attribute.
#         Input kps : (x, y, vis)
#         Output heatmap_size : (w,h) or (x,y)
#         Simple Coordinate Classification (simcc)

#         """
#         '#0 verify hyper_params'
#         if hyper_params is None:
#             self.hyper_params = self.YOLO_FPN_ANCHORS_CFG
#         else:
#             if isinstance(hyper_params, dict):
#                 '''
#                 TO DO : add keys() to verify format of hyper_params
#                 '''
#                 self.hyper_params = hyper_params
#             else:
#                 raise RuntimeError("hyper_params must be dict")    
            
#         '#1 encode part'
#         fpn_anchors_gen = PAFPN_AnchorBox(fpn_params=self.hyper_params, anchor_normalize=True)
#         self.norm_fpn_anchors_list, _ = fpn_anchors_gen(show_info=True) #[(feat_h,feat_w,3,4), .... ]
#         self.img_shape_yx = tf.constant(self.hyper_params['img_size'], dtype=self.compute_dtype) #(2,)
#         self.img_shape_xy = tf.reverse(self.img_shape_yx, axis=[0]) #(2,)

        
#         '#2 decode part'
#         self.num_classes = num_classes

#         fpn_shape_list =  self.hyper_params['feature_map_shapes'] 
#         fpn_anchor_size_list =  self.hyper_params["anchor_sizes"]  

#         assert len(fpn_shape_list)==len(fpn_anchor_size_list),"fpn_levels in fpn_shape_list and anchor_size_list must be same"
#         assert self.img_shape_xy [0]==self.img_shape_xy [1],"ONLY SUPPORT img.shape X=Y (model input)"

#         self._feat_shapes_list = fpn_shape_list                    #[80,40,20]
#         self._num_anchors_list = [ feat_shape*feat_shape*len(anchor_size_list) for feat_shape, anchor_size_list in zip(self._feat_shapes_list, fpn_anchor_size_list) ]                  
#         self._fpn_anchor_sizes = tf.constant(fpn_anchor_size_list, dtype=self.compute_dtype) #(3,3,2)
        
#         self._grid_xy_list = []
#         for feat_shape in self._feat_shapes_list: 
#             grid_coords_y = tf.cast(tf.range(0, feat_shape), dtype=self.compute_dtype)
#             grid_coords_x = tf.cast(tf.range(0, feat_shape), dtype=self.compute_dtype)
#             grid_x, grid_y = tf.meshgrid(grid_coords_x, grid_coords_y)    # (grid_coords_x,grid_height)  
#             grid_xy = tf.concat([grid_x[...,None],grid_y[...,None]], axis=-1) # (grid_coords_x,grid_height, 2) 
#             self._grid_xy_list.append(grid_xy)

#         self.input_size = tf.cast(self.img_shape_xy[0] , dtype=self.compute_dtype) # _input_size = (640,640)
#         self.num_classes = num_classes
#         self.nms_iou_threshold = nms_iou_threshold
#         self.confidence_threshold = confidence_threshold
#         self.max_detections_per_class = max_detections_per_class
#         self.max_detections = max_detections    
    

#     def Get_ANCHORS_CFG(self) -> Dict:
#         return self.hyper_params
    
#     def _gen_yolo_fpn_tragets(self, norm_fpn_anchors, norm_gt_bboxes, cls_ids):
#         #---------------------------------------------------------------
#         # norm_fpn_anchors : (feat_h, feat_w, 3, 4),  @box_type=ctr_xywh
#         # norm_gt_bboxes : (b, num_gt_bbox, 4),     @box_type=ctr_xywh
#         # cls_ids : (b, num_gt_bbox)
#         # All anchors and bboxies were normalized by img_shape_xy
#         #-----------------------------------------------------------------
#         feat_shape = norm_fpn_anchors.shape[0] #feat_h=feat_w
#         norm_bboxes_xy = norm_gt_bboxes[...,:2]   #(b,num_gt_bbox,2)
#         norm_bboxes_wh = norm_gt_bboxes[...,2:4]   #(b,num_gt_bbox,2)

        
#         '#1 mask for padding sample'
#         bboxes_prod_wh = tf.math.reduce_prod(norm_bboxes_wh, axis=-1)  #(b,num_gt_bbox,2) =>(b,num_gt_bbox)
#         eff_mask = tf.greater(bboxes_prod_wh, 0.)          #(b,num_gt_bbox) @bool

#         '#2 ctr mask'
#         gt_bboxes_xy_feat = norm_bboxes_xy*feat_shape                              #(b,num_gt_boxes,2)@feat frame, float
#         fpn_anchors_xy_feat = norm_fpn_anchors[...,:2]*feat_shape                       #(feat_h,feat_w,3,2)@feat frame, float
#         delta_ctr_xy = fpn_anchors_xy_feat[None,:,:,:,None,:] - gt_bboxes_xy_feat[:,None, None, None,:,:]  #(b,feat_h,feat_w,3,num_gt_boxes,2)
#         abs_delta_ctr_xy = tf.abs(delta_ctr_xy)                                   #(b,feat_h,feat_w,3,num_gt_boxes,2)

#         cond_ctr = tf.less_equal(abs_delta_ctr_xy, 0.5)                                    #(b,feat_h,feat_w,3,num_gt_boxes,2)
#         cond_near = tf.less_equal(abs_delta_ctr_xy, 1.)                                    #(b,feat_h,feat_w,3,num_gt_boxes,2)
#         cond_near_left_and_right = tf.stack([cond_near[...,0], cond_ctr[...,1] ], axis=-1) #(b,feat_h,feat_w,3,num_gt_boxes,2)
#         cond_near_top_and_down = tf.stack([ cond_ctr[...,0], cond_near[...,1]], axis=-1)   #(b,feat_h,feat_w,3,num_gt_boxes,2)

#         cond_ctr = tf.math.reduce_all(cond_ctr, axis=-1, keepdims=False)                                 #(b,feat_h,feat_w,3,num_gt_boxes,2) => (b,feat_h,feat_w,3,num_gt_boxes)
#         cond_near_left_and_right = tf.math.reduce_all(cond_near_left_and_right, axis=-1, keepdims=False) #(b,feat_h,feat_w,3,num_gt_boxes,2) => (b,feat_h,feat_w,3,num_gt_boxes)
#         cond_near_top_and_down = tf.math.reduce_all(cond_near_top_and_down, axis=-1, keepdims=False)     #(b,feat_h,feat_w,3,num_gt_boxes,2) => (b,feat_h,feat_w,3,num_gt_boxes)

#         ctr_mask = tf.stack([cond_ctr, cond_near_left_and_right, cond_near_top_and_down], axis=-1) #(b,feat_h,feat_w,3,num_gt_boxes)  => (b,feat_h,feat_w,3,num_gt_boxes,3)
#         ctr_mask = tf.reduce_any(ctr_mask, axis=-1)                                                #(b,feat_h,feat_w,3,num_gt_boxes,3) => (b,feat_h,feat_w,3,num_gt_boxes)                                           
#         eff_mask = tf.math.logical_and(ctr_mask, eff_mask[:,None, None, None,:])                   #(b,feat_h,feat_w,3,num_gt_boxes)***

#         '#3 wh'
#         ratios_of_gt_anchors = norm_bboxes_wh[:,None,:,:] / norm_fpn_anchors[None, 0, 0, :, None, 2:4]   #(b,3,num_gt_boxes,2)
#         ratios_of_anchors_gt = norm_fpn_anchors[None, 0, 0, :, None, 2:4] / norm_bboxes_wh[:,None,:,:]   #(b,3,num_gt_boxes,2)
#         ratios = tf.concat([ratios_of_gt_anchors,ratios_of_anchors_gt], axis=-1)                 #(b,3,num_gt_boxes,4)
#         max_ratios = tf.reduce_max(ratios, axis=-1)                                 #(b,3,num_gt_boxes)

#         '#4 combine generated targets'
#         yolo_conf = tf.where(eff_mask, max_ratios[:,None, None, :,:], 4.) #(b,feat_h,feat_w,3,num_gt_boxes)
#         eff_mask = tf.less(tf.reduce_min(yolo_conf, axis=-1), 4.)      #(b,feat_h,feat_w,3,num_gt_boxes) => (b,feat_h,feat_w,3)

#         '4-1 assign class' 
#         matched_gt_idx = tf.math.argmin(yolo_conf, axis=-1)             #(b,feat_h,feat_w,3,num_gt_boxes) => (b,feat_h,feat_w,3)
#         matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx, batch_dims=1)   #(b,num_gt_boxes) => (b,feat_h,feat_w,3)
#         cls_targets = tf.where(eff_mask, matched_gt_cls_ids, 0)          #(b,feat_h,feat_w,3)

#         '4-2 assign bbox'
#         matched_gt_boxes = tf.gather(norm_gt_bboxes, matched_gt_idx, batch_dims=1)       #(b,num_gt_boxes,4) => (b,feat_h,feat_w,3,4)
#         matched_gt_boxes = tf.where(eff_mask[...,None], matched_gt_boxes, 0.)          #(b,feat_h,feat_w,3,4)

#         '4-3 assign object confidence'
#         eff_mask = tf.cast(eff_mask, dtype=self.compute_dtype)   #(b,feat_h,feat_w,3) positive samples

#         '4-4 fpn_tragets'
#         feat_targets = tf.concat([matched_gt_boxes, cls_targets[...,None], eff_mask[...,None]], axis=-1) #(b,feat_h,feat_w,3,4+1+1)

#         return feat_targets
#     def encode(self, 
#             data : Dict[str,tf.Tensor]) ->Tuple[tf.Tensor, List[tf.Tensor]]: 
            
#         return self.batch_encode(data)
    
#     def batch_encode(self, 
#                      data : Dict[str,tf.Tensor]) ->Tuple[tf.Tensor, List[tf.Tensor]]: 
        
#         #batch_images = data['image'] 
#         gt_boxes_ctr_xywh = data['bbox'] 
#         cls_ids = data['labels'] 
#         '#0 init'
#         cls_ids = tf.cast(cls_ids, dtype=self.compute_dtype)
#         norm_gt_bboxes = gt_boxes_ctr_xywh/tf.tile(self.img_shape_xy,(2,))           #(b,num_gt_boxes,4)
#         feat_targets_list = []
#         for norm_fpn_anchors in self.norm_fpn_anchors_list:
#             feat_targets = self._gen_yolo_fpn_tragets(norm_fpn_anchors, norm_gt_bboxes, cls_ids)
#             feat_targets_list.append(feat_targets)                                   # [(b,feat_h,feat_w,3,6),.... ]

#         'invalid type of data value in dict must be tuple[tf.Tensor] or tf.Tensor not list'    
#         data['y_true'] = (*feat_targets_list,)
#         #return batch_images, (*feat_targets_list,)
#         return data
    

#     def decode(self, 
#                y_preds_list : tf.Tensor) -> tf.Tensor:
#         return self.batch_decode(y_preds_list)
    
#     def batch_decode(self, 
#                y_preds_list : List[tf.Tensor]) -> object:
        
#         """
#         calculate Loss for per output features
#         y_pred: (batch_size, 80,80,3, 4+num_cls+1)/ (batch_size, 40,40,3, 6) /(batch_size, 20,20,3, 6) from Model output
#         #Note : Here, 4+1+1 means [ bbox_xywh, cls, Object_Mask]
#         """
#         cls_preds = []
#         box_preds = []
#         for fpn_level, y_pred in enumerate(y_preds_list) :

#             'y_pred in one feat'
#             reg_pred_i = y_pred[...,:4]            #(b,feat_h,feat_w,3,4)
#             cls_pred_i = y_pred[...,4:4+self.num_classes]  #(b,feat_h,feat_w,3,mum_cls)
#             obj_pred_i = y_pred[...,-1]            #(b,feat_h,feat_w,3)
            
#             'decode predictions'
#             grid_xy = self._grid_xy_list[fpn_level]   #(feat_h,feat_w,2) 
#             feat_shapes = tf.cast( self._feat_shapes_list[fpn_level], dtype=self.compute_dtype)  #(,) 
#             anchor_size_3x2 = self._fpn_anchor_sizes[fpn_level,...] #(3,2)
#             #print(reg_pred_i.shape, grid_xy.shape, feat_shapes)
#             box_xy = (tf.nn.sigmoid(reg_pred_i[..., :2])*2-0.5 + grid_xy[None,:,:,None,:])/feat_shapes #(b,feat_h,feat_w,3,2)
#             box_wh = ( (tf.nn.sigmoid(reg_pred_i[..., 2:4])*2)**2 )*anchor_size_3x2[None, None, None,:,:]/self.input_size #(b,feat_h,feat_w,3,2)
#             box_pred_i = tf.concat([box_xy,box_wh], axis=-1) #(b,feat_h,feat_w,3,4)

#             'cls prediction'
#             cls_pred_i = tf.nn.sigmoid(cls_pred_i)
#             obj_pred_i = tf.nn.sigmoid(obj_pred_i)
#             cls_pred_i  = cls_pred_i*obj_pred_i[...,None]  #(b,feat_h,feat_w,3,mum_cls)

#             'reshape'
#             cls_pred_i = tf.reshape(cls_pred_i, shape=(-1, self._num_anchors_list[fpn_level], self.num_classes))   #(b,feat_h*feat_w*3,mum_cls)
#             box_pred_i = tf.reshape(box_pred_i, shape=(-1, self._num_anchors_list[fpn_level], 4))

#             cls_preds.append(cls_pred_i)
#             box_preds.append(box_pred_i)

#         cls_preds =  tf.concat(cls_preds , axis=1)            #(b,25200,4)
#         box_preds =  tf.concat(box_preds , axis=1)
#         box_preds = tf.expand_dims(box_preds, axis=2)          #(b,25200,1,4)
#         box_preds = self.bboxes_centers2corners(box_preds)
#         box_preds = tf.tile(box_preds,[1,1,4,1])*self.input_size     #(b,25200,4,4)  

        
#         'Decode bbox and cls predictions by tf.image.combined_non_max_suppression'
#         # https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
#         return tf.image.combined_non_max_suppression(
#                                 box_preds,
#                                 cls_preds,
#                                 self.max_detections_per_class,
#                                 self.max_detections,
#                                 self.nms_iou_threshold,
#                                 self.confidence_threshold,
#                                 clip_boxes=False,
#                             )    
     
#     def bboxes_centers2corners(self,
#                             boxes :Tensor) -> Tensor:
#         '''
#         cxcy,wh ->yx_min, yx_max
#         [cx, cy]-[w, h]/2 = x_min, y_min
#         [cx, cy]+[w, h]/2 = x_max, y_max
#         '''
#         boxes = tf.concat([boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],axis=-1) #xy_min,xy_max
#         boxes = tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1) # yx_min, yx_max  
#         return boxes    


    

