import tensorflow as tf
from lib.datasets.transforms import BBoxesFormatTransform
from lib.codecs.bounding_box import compute_ious,  dist2bbox, get_anchor_points, is_anchor_center_within_box
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from lib.codecs import BaseCodec
from lib.Registers import CODECS
#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
#https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tal.py
@CODECS.register_module()
class YoloAnchorFreeCodec(BaseCodec):
    VERSION = '1.0.0'
    ENCODER_USE_PRED = True
    ENCODER_GEN_SAMPLE_WEIGHT = False
   
    r""" task assignment(enocde) and prediction decode, TaskAlignedAssigner
    Author : Dr. David Shen
    Date : 2024/2/23
    
    Encodes ground truth boxes to target boxes and class labels for training a
    YOLOV8 model. This is an implementation of the Task-aligned sample
    assignment scheme proposed in https://arxiv.org/abs/2108.07755.

    requied function : get_anchor_points, is_anchor_center_within_box, dist2bbox

    References:
        - [Based on implementation of 'TaskAlignedAssigner' @ultralytics] (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tal.py)
        - [Inspired on 'YOLOV8LabelEncoder' @keras-cv] (https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/object_detection/yolo_v8/yolo_v8_label_encoder.py)
        - [Inspired on 'Loss' @leondgarse] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/yolov8/losses.py#L329)
    Args:
        num_classes (int) : The number of object classes.
        image_shape(Tuple[int]) :  tuple or list of two integers representing the height and
            width of input images, respectively. Model's input shape. Defaults to (640,640)
        max_anchor_matches (int) :  topk, The number of top candidates to consider, Defaults to 10.
        strides (int) : tuple of list of integers, the size of the strides across the
                image size that should be used to create anchors. Defaults to [8, 16, 32].
        reg_max (int) : The maximum regression channels used in decode_regression_to_boxes. Defaults to 16
        apply_df_loss (bool) :  Whether to apply distribution_focal_loss in training. Defaults to True.
                if Ture, pred_bbox.shape =(b, 8400,4+reg_max*4), 
                if False,  pred_bbox.shape =(b, 8400,4)
        alpha (float) :   a parameter to control the influence of class predictions on the alignment score of an anchor box. 
            This is the alpha parameter in equation 9 of https://arxiv.org/pdf/2108.07755.pdf. Defaults to 0.5 .
        beta (float): The beta parameter for the localization component of the task-aligned metric. Defaults to 6..
                a parameter to control the influence of box IOUs on the alignment score of an anchor box. 
                This is the beta parameter in equation 9 of https://arxiv.org/pdf/2108.07755.pdf.
        epsilon (float) : small value to prevent division by zero. , defaults to 1e-9.
        iou_type (str) : which iou type to be appled for computing overlap. Defaults to "ciou"
        gt_bbox_format (str): the format of bounding boxes of input dataset. Defaults to 'xywh'

    Note :
       - with_auxiliary_dfl_regression=True , support apply df_loss in training , y_pred_bbox.shape=(b, 8400, 4+4*reg_max)
       - with_auxiliary_dfl_regression=False , y_pred_bbox.shape=(b, 8400, 4), only use ciou_box_loss in training
    
    Example:
        '''Python
        codec = YoloAnchorFreeCodec(num_classes = 4,
                            image_shape = (640,640),
                            max_anchor_matches=10,
                            strides=[8, 16, 32],
                            with_auxiliary_dfl_regression = True,
                            alpha=0.5,
                            beta=6.0,
                            epsilon=1e-9,
                            cls_label_shift = -1,
                            gt_bbox_format = 'xywh')  
    """
    def __init__(self,
                num_classes,
                image_shape : Tuple[int]=(640,640),
                max_anchor_matches : int = 10,
                strides: Union[Tuple[int], List[int] ] = [8, 16, 32],
                reg_max  : int = 16,
                with_auxiliary_dfl_regression : bool = True,
                alpha : float = 1.0,
                beta : float = 6.0,
                epsilon : float= 1e-9,
                cls_label_shift : int =  -1,
                iou_type : str = 'ciou',
                gt_bbox_format :str = 'xywh', 
                **kwargs):
        super().__init__(**kwargs)
    
        "basic cfg"
        self.num_classes = num_classes
        self.cls_label_shift = cls_label_shift
        self.max_anchor_matches = max_anchor_matches
        self.reg_max = reg_max
        self.with_auxiliary_dfl_regression = with_auxiliary_dfl_regression
        self.alpha = tf.cast( alpha, dtype=self.compute_dtype)
        self.beta = tf.cast( beta, dtype=self.compute_dtype)
        self.epsilon = tf.cast( epsilon, dtype=self.compute_dtype)
        self.strides = strides
        self.image_shape = image_shape


        '''
        anchor_points : (8400,2)
        anchor_srides : (8400,1)
        '''
        self.anchor_points, self.anchor_srides = get_anchor_points(
            self.image_shape, self.strides, self.compute_dtype
        )
        self.num_anchors = self.anchor_points.shape[0] #8400=80**2 +40**2 +20**2
        print(f"anchor_points : {self.anchor_points.shape} , anchor_srides : {self.anchor_srides.shape}")
        'init bbox format conversion'
        if gt_bbox_format != 'xyxy':
            self.gt_bboxes_format_transform = BBoxesFormatTransform(
                convert_type=gt_bbox_format+'2xyxy'
            )
            print(f"apply bboxes_format_transform <{gt_bbox_format+'2xyxy'}> ")

        'init compute_ious'
        self.compute_ious = compute_ious(
                    mode = iou_type,
                    boxes1_src_format ='xyxy',
                    boxes2_src_format ='xyxy',
                    is_aligned=False,
                    use_masking=False
        )

        'cfg for decoder'
        self.decoder_bboxes_format_transform = BBoxesFormatTransform(
            convert_type='xyxy'+'2yxyx'
        )

    def get_box_metrics(
            self, 
            pred_scores, 
            decode_bboxes,
            gt_labels, 
            gt_bboxes, 
            mask_gt
    ) ->Tuple[tf.Tensor]:
        """
        cls_num is number of object class, voc_cls_num=20 , coco_cls_num=80
        pred_scores : (b,8400,cls_num)
        decode_bboxes : (b,8400,4) , xyxy format @img_frame
        gt_labels : (b,num_gt_bbox)
        gt_bboxes : (b,num_gt_bbox,4), xyxy format @img_frame
        mask_gt ( 1.or 0. float): (b,num_gt_bbox,1)
        """
     
        bbox_scores = tf.gather(
                pred_scores,
                tf.math.maximum(tf.cast(gt_labels, dtype=tf.int32),[0]),
                batch_dims=-1
        )#(b,h*w,20)=> (b,h*w,num_gt_bbox),  tf.gather cannot support index=-1, 
        #so if lable=-1, it will always return value of pred_scores[b,0]
        bbox_scores = tf.transpose(
                bbox_scores, perm=[0, 2, 1]
        )#(b,h*w,num_gt_bbox)=>(b,num_gt_bbox,h*w)
        bbox_scores *= mask_gt
        #(b,num_gt_bbox,h*w)*(b,num_gt_bbox,1) => (b,num_gt_bbox,h*w)
        # when lable=-1,  bbox_scores get value of  pred_scores[b,0]
        # use mask_gt to zero out these invalid values
        overlaps =  self.compute_ious(gt_bboxes, decode_bboxes)
        #(b,num_gt_bbox,h*w)
        overlaps *= mask_gt
   
        #(b,num_gt_bbox,h*w)*(b,num_gt_bbox,1) => (b,num_gt_bbox,h*w)
        overlaps = tf.math.maximum(overlaps,0.)
        #(b,num_gt_bbox,h*w)
        alignment_metrics = tf.math.pow(bbox_scores,self.alpha)*tf.math.pow(overlaps,self.beta)
        #(b,num_gt_bbox,h*w)*(b,num_gt_bbox,h*w)=>(b,num_gt_bbox,h*w)
        # print('align_metrics', alignment_metrics)
        # print('overlaps', overlaps)

        return alignment_metrics, overlaps

    def select_topk_candidates(self, alignment_metrics, mask_gt):
        #self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        """

        alignment_metrics : (b, num_gt_bbox, 8400)
        mask_gt : (b,num_gt_bbox,1) @float [0. or 1.]
        """
        candidate_metrics, candidate_idxs = tf.math.top_k(
            alignment_metrics, self.max_anchor_matches
        ) #(b,num_gt_bbox,8400) => metrics:(b,num_gt_bbox,10),[0~1] & idxs:(b,num_gt_bbox,10) [0~8399]
        # if all elements in axis=-1 are zeros, top_k will give invalid idx=0 and metric=0
        # , so we need to use mask_gt to filter out these idx
   
        candidate_idxs = tf.where(
            tf.cast(mask_gt, dtype=tf.bool), candidate_idxs, -1
        ) #(b,num_gt_bbox,8400) 
        candidate_idxs = tf.where(
            tf.greater(candidate_metrics,0.), candidate_idxs, -1
        )# it may be removed if already overlaps.clamp(0)
        mask_topk = tf.one_hot(
            candidate_idxs,self.num_anchors,axis=-1
        ) #(b,num_gt_bbox,10,num_anchors=8400) 
    
        #tf.one_hot([-1,0,1],depth=3) will return [[0,0,0], [1,0,0], [0,1,0]]
        mask_topk = tf.math.reduce_sum(mask_topk, axis=-2) 
        #(b,num_gt_bbox,10,8400) => (b,num_gt_bbox,8400)
        return mask_topk

    def select_highest_overlaps(self,overlaps, mask_pos):
        """
        Args:
        mask_pos (Tensor): shape(b, num_gt_bbox, h*w)
        overlaps (Tensor): shape(b, num_gt_bbox, h*w)

        Returns:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        """
        fg_mask = tf.math.reduce_sum(mask_pos, axis=-2) 
        # mask_pos : (b,num_gt_bbox,h*w) => fg_mask : (b,h*w) 
        mask_multi_gts = tf.greater(fg_mask,1) 
        # mask_multi_gts : (b,h*w) @boolean tensor
        # to find elelment at axis=-1 is greater than 1
        max_overlaps_idx = tf.math.argmax(overlaps, axis=1)  
        # (b,n_max_boxes, h*w) => (b,h*w)
        is_max_overlaps = tf.one_hot(
            max_overlaps_idx, tf.shape(overlaps)[1], dtype=self.compute_dtype
        ) #(b,h*w,num_gt_bbox)
        is_max_overlaps = tf.transpose(
            is_max_overlaps,(0,2,1)
        ) #(b,num_gt_bbox, h*w)
        mask_pos = tf.where(
            mask_multi_gts[:,None,:], is_max_overlaps, mask_pos
        )#(b,num_gt_bbox, h*w)
        # each col in mask_pos should have one element that is 1 
        fg_mask = tf.math.reduce_sum(
            mask_pos, axis=-2
        ) #(b,num_gt_bbox, h*w) => (b,h*w)
        # target_gt_idx = tf.where(tf.cast(fg_mask, dtype=tf.bool),
        #               tf.math.argmax(mask_pos, axis=-2),
        #               -1)      # (b,num_gt_bbox, 8400)->(b,8400) [-1,-1,0,1,1,2]
        '''
        Note : finnally mask_pos should be as following
            each col should have only one element=1 most
            but some cols present all elemenet is zeros
            in this condition, tf.math.argmax(mask_pos, axis=-2) will out target_gt_idx=0 
            obviously, it's invalid, so we need to use fg_mask to filter out these invalid element
                anchors(h*w=6)
                0 1 2 3 4 5
            gt_idx=0  [1 0 0 0 0 0]
            gt_idx=1  [0 1 1 0 0 0]
            gt_idx=2  [0 0 0 1 1 0]

        '''
        target_gt_idx = tf.math.argmax(mask_pos, axis=-2)
        # target_gt_idx : (b, h*w)
        #
        #
        return target_gt_idx, fg_mask, mask_pos
    
    def assign(
        self, gt_bboxes, gt_labels, mask_gt, pred_bboxes, pred_scores
    ):
        '#2-----------------------get pos_mask / align_metrics / overlaps  --------------------------'
        
        mask_in_gt_boxes = is_anchor_center_within_box(
            self.anchor_points*self.anchor_srides, gt_bboxes
        ) #(b,num_gt_bbox,8400) @boolean tensor
    

        align_metrics, overlaps = self.get_box_metrics(
                                pred_scores = pred_scores,
                                decode_bboxes = pred_bboxes,
                                gt_labels = gt_labels,
                                gt_bboxes = gt_bboxes,
                                mask_gt = mask_gt*mask_in_gt_boxes
        )# align_metrics : (b,num_gt_bbox,8400), overlaps : (b,num_gt_bbox,8400) 
        # overlaps and align_metrics both are between 0.~1.
 
        mask_topk = self.select_topk_candidates(align_metrics, mask_gt)
        
        mask_pos = mask_topk*mask_in_gt_boxes*mask_gt
        
        
        '#3 ---------get target_gt_idx / fg_mask / mask_pos (some anchors were assigned by two labels) ----'
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(overlaps, mask_pos)


        '#4 ----- get target, bbox@image_frame----------------------------------'
        # fg_mask : (b, h*w) @float [0. or 1.]
        # mask_pos : (b, num_gt_bbox, h*w)  @float [0. or 1.]
        
        target_bboxes = tf.gather(
            gt_bboxes, target_gt_idx,batch_dims=1
        )#(b,num_gt_bbox,4)=>(b,8400,4)
        target_bboxes *= fg_mask[...,None]
        #(b,8400,4)
        # target_bbox = tf.where(
        #     fg_mask[...,None], target_bbox, 0.
        # )#(b,8400,4)


        target_scores = tf.gather(
            gt_labels, target_gt_idx, batch_dims=1
        ) # target_scores : (b,num_gt_bbox)=>(b,8400)  [-1,num_classes-1]
 
        target_scores = tf.where(
            tf.cast(fg_mask, dtype=tf.bool), target_scores, -1
        )#(b,8400)   ************************************
        target_scores = tf.one_hot(
            target_scores, self.num_classes, axis=-1, dtype=self.compute_dtype
        ) #(b,8400)=> #(b,8400,20)
      
       
        align_metrics *= mask_pos #(b,num_gt_bbox,8400)*(b,num_gt_bbox,8400) => (b,num_gt_bbox,8400)
        overlaps *= mask_pos
        
        # print('pos_overlaps', overlaps)
        # print('pos_align_metrics', align_metrics)
        # print('target_scores', target_scores)
        # print( "target_scores_sum : " , tf.reduce_sum(target_scores, axis=[1,2]))
        # print("valid target_bboxes : ", tf.boolean_mask(target_bboxes, fg_mask, axis=0, name='boolean_mask'))
        '#5 --------------- normalize targets : target_scores & target_bbox--------------'
        pos_align_metrics  = tf.math.reduce_max(
            align_metrics, axis=-1,keepdims=True
        ) # (b,num_gt_bbox,8400)=>(b,num_gt_bbox,1)
        pos_overlaps  = tf.math.reduce_max(
            overlaps, axis=-1, keepdims=True
        ) #(b,num_gt_bbox,8400) => (b,num_gt_bbox,1)
        #print(target_scores.dtype, align_metrics.dtype, pos_overlaps.dtype, pos_align_metrics.dtype)
        norm_align_metrics = align_metrics*pos_overlaps/(pos_align_metrics + self.epsilon) #(b,num_gt_bbox,8400)
        norm_align_metrics = tf.math.reduce_max(
            norm_align_metrics, axis=1
        ) #(b,num_gt_bbox,8400)=> #(b,8400)
        
        target_scores *= norm_align_metrics[:,:,None]
        #bbox_labels = tf.reshape(bbox_labels, (-1, self.num_anchors, 4))

        return (
            tf.stop_gradient(target_bboxes),
            tf.stop_gradient(target_scores),
            tf.stop_gradient(fg_mask),
            tf.stop_gradient(target_gt_idx)
        )

    def batch_encode(self, data, y_pred):
        r""" 
        args :
            data (dict[str,Tensor]) :
                gt_data that should contain "labels" and "bbox" keys

                - data["labels"] (int,[b,num_gt_bbox], [0~num_cls] ) :
                                gt_labels,  dtype=tf.int32
                                tensor value are between 0~80 @coco_dataset with padding value = 0, 
                                so valid labels are 1~80
                - data["bbox"] (float,[b,h*w,4],  [0~640.], xyxy@image_frame)  : 
                                gt_bbox with xywh format bbox @image_frame, dtype=self.compute_dtype
                                tensor value are between 0~640. @img_size=640 with padding value = 0.

            y_pred (Tuple[Tensor]):
                model's prediction (pred_distri_bboxes, pred_scores)
                - pred_distri_bboxes (float, [b, h*w, 4],  [[0~80.],[0~40], [0,20]], xyxy@feat_frame) ) : 
                            xyxy format @feature_frame, distance to anchor points
                            we will transform it to pred_bboxes xyxy@feature_frame
                - pred_scores (float, [b, h*w, num_cls], [0~1]): 
                            each cls score after simoid, each anchor point has num_cls
        

        process :

                mask_gt (float, [b, h*w],  [0.|1.])  : 
                mask_pos (float, [b, num_gt_bbox, h*w],  [0.|1.]) :  

        return :
            data (dict[str,Tensor]) :
                data['y_pred'] =  (pred_bboxes, pred_scores)
                data['y_true'] =  (target_bboxes, target_scores)
                data['sample_weight'] = (box_weight, cls_weight)
        
                pred_bboxes (float, [b, h*w, 4], [[0~80.],[0~40], [0,20]], xyxy@feat_frame ):
                pred_scores (float, [b, h*w, 4], [0~1] ):
                target_bboxes (float, [b, h*w, 4], [[0~80.],[0~40], [0,20]], xyxy@feat_frame ):
                target_scores (float, [b, h*w, 4], [0.|1.]):
                box_weight (float, [b, h*w, 4] ):
                cls_weight (float, [b, h*w, 4] ):

       
        pred_bboxes (float) : xyxy @feature frame after dist2bbox
        gt_bboxes (float) : xywh @image frame
        target_bboxes (float): xyxy @feature frame
        mask_gt(float)  : (b,h*w,1) @[0. or 1.]
        mask_pos(float) : (b,num_gt_bbox, h*w) @[0. or 1.]
        """
    

        pred_distri_bboxes_reg, pred_scores = y_pred
        # pred_distri_bboxes_reg = tf.cast(pred_distri_bboxes_reg, dtype=self.compute_dtype)
        # pred_scores = tf.cast(pred_scores, dtype=self.compute_dtype)



        pred_distri_bboxes = pred_distri_bboxes_reg[...,:4]
        # #print(pred_distri_bboxes_reg.dtype, pred_scores.dtype, data['bbox'].dtype)
        # pred_distri_bboxes = tf.cast(pred_distri_bboxes_reg[...,:4], dtype=self.compute_dtype)
        # pred_scores = tf.cast(pred_scores, dtype=self.compute_dtype)

        
        pred_bboxes = dist2bbox(
            pred_distri_bboxes, self.anchor_points
        ) #(b,8400,4) @feature_frames
       
        gt_labels = data["labels"] + self.cls_label_shift #cls_ids
        # gt_labels : (b,num_gt_bbox) , in coco dataset(80 class), lables is 0~80 with valid label=1~80 and padding=0 
        # by cls_label_shift => lables becomes -1~79  with valid label=0~79 and padding=-1
        # i.e : gt_labels with cls_label_shift=-1 [1,2,3,4,0,0] => [0,1,2,4,-1,-1] cls_label_shift=-1

        gt_bboxes = data["bbox"] #(b,num_gt_bbox,4) xywh @image frame
        if hasattr(self,'gt_bboxes_format_transform'):
            gt_bboxes = self.gt_bboxes_format_transform(gt_bboxes) #(b,num_gt_bbox,4) xywh=>xyxy  @image_frames
      
        '#1. ---------------------get mask_gt---------------------------------------------------'
        #mask_gt = tf.reduce_all(tf.greater(gt_bboxes,0.), axis=-1, keepdims=True) #(b,num_gt_bbox,1) ?????
        #mask_gt = tf.greater(tf.math.reduce_max(gt_bboxes, axis=-1,keepdims=True),0.) #(b,num_gt_bbox,4)=> (b,num_gt_bbox,1)
        mask_gt = tf.cast( 
            tf.greater_equal(gt_labels[...,None],0) , dtype=self.compute_dtype
        )#(b,num_gt_bbox,)=> (b,num_gt_bbox,1)
        '# --------------- assign task(get pos_mask / align_metrics / overlaps ) ----------------'
        target_bboxes, target_scores, fg_mask, _ = self.assign(
            gt_bboxes, 
            gt_labels, 
            mask_gt, 
            pred_bboxes*self.anchor_srides, 
            pred_scores
        )

        target_bboxes /= self.anchor_srides
        '#5 ----------- out target, bbox@feature_frame--------------------------------------------------'
        
        #target_scores  : (b, h*w, num_cls)
        target_scores_sum = tf.math.maximum(tf.math.reduce_sum(target_scores), 1.) # all valid anchor points
        box_weight = tf.math.reduce_sum(target_scores, axis=-1)*fg_mask # 應該可以刪除 (b,num_anchors)
        box_weight = box_weight[...,None]/target_scores_sum
        cls_weight = 1./target_scores_sum

        pred_bboxes = pred_bboxes*fg_mask[...,None] 
        if self.with_auxiliary_dfl_regression :
            pred_distri_reg =  pred_distri_bboxes_reg[...,4:]
            if pred_distri_reg.shape[-1]!= self.reg_max*4:
                raise ValueError(
                    " IF with_auxiliary_dfl_regression = True, "
                    " codec.encoder expects number of channels in y_pred[...,4:].shape[-1] to be equal to reg_max*4"
                    f" Received number of channels in y_pred[...,4:].shape[-1] is {pred_distri_reg.shape[-1]} "
                    f" expected number of channels is {self.reg_max*4}"
                )
            data['y_pred'] =  (tf.concat([pred_bboxes, pred_distri_reg], axis=-1) ,  pred_scores)
        else:
            data['y_pred'] =  (pred_bboxes ,  pred_scores)

        data['y_true'] =  (target_bboxes, target_scores)
        data['sample_weight'] = (box_weight, cls_weight)
        return  data
    
    def decode_regression_to_boxes(self, pred_distri_reg_bboxes):
        """Decodes the results of the YOLOV8Detector forward-pass into boxes.

        Returns left / top / right / bottom predictions with respect to anchor
        points.

        Each coordinate is encoded with 16 predicted values. Those predictions are
        softmaxed and multiplied by [0..15] to make predictions. The resulting
        predictions are relative to the stride of an anchor box (and correspondingly
        relative to the scale of the feature map from which the predictions came).
        """

        pred_distri_reg_bboxes = tf.keras.layers.Reshape((-1, 4, self.reg_max))(
            pred_distri_reg_bboxes
        )
        pred_distri_reg_bboxes = tf.nn.softmax(
            pred_distri_reg_bboxes, axis=-1, name='softmax'
        )*tf.range(start=0., limit=self.reg_max, delta=1.)

        return tf.math.reduce_sum(
            pred_distri_reg_bboxes, axis=-1
        )
    
    def batch_decode(
        self, y_pred,  meta_data : Optional[dict] = None, *args, **kwargs
    ):
        r""" 
        tf.image.combined_non_max_suppression(
                    boxes,
                    scores,
                    max_output_size_per_class = 10,
                    max_total_size = 10,
                    iou_threshold=0.5,
                    score_threshold=float('-inf') = 0.5,
                    pad_per_class=False,
                    clip_boxes=True,
                    name=None
            )  
        """

        pred_distri_bboxes_reg, pred_scores = y_pred
        if pred_distri_bboxes_reg.shape[-1]==self.reg_max*4 :
            pred_distri_reg =  tf.cast(
                pred_distri_bboxes_reg[...,4:], dtype=self.compute_dtype
            )
            pred_distri_bboxes = self.decode_regression_to_boxes(
                pred_distri_reg
            )
        else:
            pred_distri_bboxes = tf.cast(
                pred_distri_bboxes_reg[...,:4], dtype=self.compute_dtype
            )
        pred_scores = tf.cast(
            pred_scores, dtype=self.compute_dtype
        ) #(b,8400,num_cls) 
        pred_bboxes = dist2bbox(
            pred_distri_bboxes, self.anchor_points
        ) #(b,8400,4) @feature_frames

        pred_bboxes= pred_bboxes*self.anchor_srides #(b,8400,4) @image_frames
        pred_bboxes = self.decoder_bboxes_format_transform(pred_bboxes)
        pred_bboxes = tf.expand_dims(pred_bboxes, axis=2)          #(b,25200,1,4)
        pred_bboxes = tf.tile(pred_bboxes,[1,1,4,1])    #(b,25200,4,4)  

        return tf.image.combined_non_max_suppression(
                    pred_bboxes,
                    pred_scores,
                    clip_boxes=False,
                    max_output_size_per_class = 10,
                    max_total_size = 10,
                    iou_threshold = 0.7,
                    score_threshold = 0.5
        ) 
    
#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
@CODECS.register_module()
class YoloPoseCodec(YoloAnchorFreeCodec):
    VERSION = '1.0.0'
    ENCODER_USE_PRED = True
    ENCODER_GEN_SAMPLE_WEIGHT = False
    r"""YoloPoseCodec used in yolov8
    Author : Dr. David Shen
    Date : 2024/2/26

    https://github.com/ultralytics/ultralytics/blob/0572b294458524f3861bb4af6204150eaf47852b/ultralytics/utils/loss.py#L147
    """
   

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    def assign_kpts(
            self, gt_kpts, fg_mask, target_gt_idx , target_bboxes
    ):
        r""" 
        gt_kpts : (b,num_gt_bbox,17,3)
        mask_gt : (b,num_anchors)
        target_gt_idx : (b,num_anchors)
        target_bboxes : (b,num_anchors,4)
        
        """
     
        'target_kpts'
        target_kpts  = tf.gather(
            gt_kpts, target_gt_idx, batch_dims=1
        ) #(b,num_anchors,17,3)
        target_kpts = tf.concat(
            [target_kpts[...,:2]/self.anchor_srides[None,:,:,None], target_kpts[...,2:3]],
            axis = -1
        )  #(b,num_anchors,17,3)
        target_kpts *= fg_mask[:,:,None, None] #(b,num_anchors,17, 3)

        'area'
        area = tf.math.reduce_prod(
            target_bboxes[...,2:]-target_bboxes[...,:2], axis=-1
        ) #(b,num_anchors,2) => (b,num_anchors)

        'final target_kpts'
        r''' final target_kpts
        i.e. :  target_kpts[0,0,...] =
            [
                [15. 17., 300.]
                [0. 0., 0.]
                [5. 7., 300.]
                [11. 12., 300.]
                [0. 0., 0.]
                    :
                [0. 0., 0.]
            ]
        here, area = 300 for this object
        '''
        target_kpts = tf.concat( 
            [
                target_kpts[...,:2],
                tf.where( 
                    target_kpts[...,2:3] >0., 
                    area[..., None,None], 
                    0.)
            ],
            axis=-1
        )# (b,num_anchors,17,2) + (b,num_anchors,17,1) => (b,num_anchors,17,3)
        return tf.stop_gradient(target_kpts)
        
    
    def kpts_decode(self,pred_kpts, anchor_points):
        """Decodes predicted keypoints to image coordinates."""
        # pred_kpts : (b, num_anchors, 17, 3)
        # anchor_points : (num_anchors, 2)
        pred_kpts_xy =  2.0 * pred_kpts[...,:2]

        return tf.concat(
            [pred_kpts_xy+anchor_points[None,:,None,:]-0.5, pred_kpts[...,2:3]],
            axis = - 1
        )
    
    def batch_encode(self, data, y_pred):
        r""" 

        """
        pred_distri_bboxes_reg, pred_scores, pred_distri_kpts = y_pred
        pred_distri_bboxes =  pred_distri_bboxes_reg[...,:4]
        'bbox decode'
        pred_bboxes = dist2bbox(
            pred_distri_bboxes, self.anchor_points
        ) #(b,8400,4) @feature_frames
        'kpts decode'
        pred_kpts = self.kpts_decode(
            pred_distri_kpts, self.anchor_points
        )

        gt_labels = data["labels"] + self.cls_label_shift #cls_ids
        # gt_labels : (b,num_gt_bbox) , in coco dataset(80 class), lables is 0~80 with valid label=1~80 and padding=0 
        # by cls_label_shift => lables becomes -1~79  with valid label=0~79 and padding=-1
        # i.e : gt_labels with cls_label_shift=-1 [1,2,3,4,0,0] => [0,1,2,4,-1,-1] cls_label_shift=-1

        gt_bboxes = data["bbox"] #(b,num_gt_bbox,4) xywh @image frame
        if hasattr(self,'gt_bboxes_format_transform'):
            gt_bboxes = self.gt_bboxes_format_transform(gt_bboxes) #(b,num_gt_bbox,4) xywh=>xyxy  @image_frames
      
        '#1. ---------------------get mask_gt---------------------------------------------------'
        #mask_gt = tf.reduce_all(tf.greater(gt_bboxes,0.), axis=-1, keepdims=True) #(b,num_gt_bbox,1) ?????
        #mask_gt = tf.greater(tf.math.reduce_max(gt_bboxes, axis=-1,keepdims=True),0.) #(b,num_gt_bbox,4)=> (b,num_gt_bbox,1)
        mask_gt = tf.cast( tf.greater_equal(gt_labels[...,None],0) , dtype=self.compute_dtype)#(b,num_gt_bbox,)=> (b,num_gt_bbox,1)
        '# --------------- assign task(get pos_mask / align_metrics / overlaps ) ----------------'
        target_bboxes, target_scores, fg_mask, target_gt_idx = self.assign(
            gt_bboxes, 
            gt_labels, 
            mask_gt, 
            pred_bboxes*self.anchor_srides, 
            pred_scores
        )
        target_bboxes /= self.anchor_srides

        '#5 ----------- keypoints-------------------------------------------------'
        gt_kpt = data["kps"] #(b, num_bboxes,17,3)

        target_kpts = self.assign_kpts(
            gt_kpts = gt_kpt,  
            fg_mask = fg_mask, 
            target_gt_idx = target_gt_idx, 
            target_bboxes = target_bboxes
        )#(b,num_anchors,17,4)

        '#6 ----------- out target, bbox@feature_frame--------------------------------------------------'
        #target_scores  : (b, h*w, num_cls)
        target_scores_sum = tf.math.maximum(tf.math.reduce_sum(target_scores), 1.) # all valid anchor points
        box_weight = tf.math.reduce_sum(target_scores, axis=-1)*fg_mask # 應該可以刪除 (b,num_anchors)
        box_weight = box_weight[...,None]/target_scores_sum
        cls_weight = 1./target_scores_sum
        kpts_weight = 1./target_scores_sum

        pred_bboxes = pred_bboxes*fg_mask[...,None] 
        pred_kpts = pred_kpts*fg_mask[:,:,None, None] #(b,num_anchors,17,3)
        if self.with_auxiliary_dfl_regression :
            pred_distri_reg =  pred_distri_bboxes_reg[...,4:]
            if pred_distri_reg.shape[-1]!= self.reg_max*4:
                raise ValueError(
                    " IF with_auxiliary_dfl_regression = True, "
                    " codec.encoder expects number of channels in y_pred[...,4:].shape[-1] to be equal to reg_max*4"
                    f" Received number of channels in y_pred[...,4:].shape[-1] is {pred_distri_reg.shape[-1]} "
                    f" expected number of channels is {self.reg_max*4}"
                )
            data['y_pred'] =  (tf.concat([pred_bboxes, pred_distri_reg], axis=-1), pred_scores, pred_kpts)

        else:
            data['y_pred'] =  (pred_bboxes ,  pred_scores , pred_kpts)

        data['y_true'] =  (target_bboxes, target_scores, target_kpts)
        data['sample_weight'] = (box_weight, cls_weight, kpts_weight)
        return  data
    

    
# #https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tal.py
# @CODECS.register_module()
# class YoloAnchorFreeCodec:
#     VERSION = '1.0.0'
#     r""" task assignment(enocde) and prediction decode, TaskAlignedAssigner
#     Encodes ground truth boxes to target boxes and class labels for training a
#     YOLOV8 model. This is an implementation of the Task-aligned sample
#     assignment scheme proposed in https://arxiv.org/abs/2108.07755.

#     requied function : get_anchor_points, is_anchor_center_within_box, dist2bbox

#     References:
#         - [Based on implementation of 'TaskAlignedAssigner' @ultralytics] (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tal.py)
#         - [Inspired on 'YOLOV8LabelEncoder' @keras-cv] (https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/object_detection/yolo_v8/yolo_v8_label_encoder.py)
#         - [Inspired on 'Loss' @leondgarse] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/yolov8/losses.py#L329)
#     Args:
#         num_classes (int) : The number of object classes.
#         image_shape(Tuple[int]) :  tuple or list of two integers representing the height and
#             width of input images, respectively. Model's input shape. Defaults to (640,640)
#         max_anchor_matches (int) :  topk, The number of top candidates to consider, Defaults to 10.
#         strides (int) : tuple of list of integers, the size of the strides across the
#                 image size that should be used to create anchors. Defaults to [8, 16, 32].
#         reg_max (int) : The maximum regression channels used in decode_regression_to_boxes. Defaults to 16
#         apply_df_loss (bool) :  Whether to apply distribution_focal_loss in training. Defaults to True.
#                 if Ture, pred_bbox.shape =(b, 8400,4+reg_max*4), 
#                 if False,  pred_bbox.shape =(b, 8400,4)
#         alpha (float) :   a parameter to control the influence of class predictions on the alignment score of an anchor box. 
#             This is the alpha parameter in equation 9 of https://arxiv.org/pdf/2108.07755.pdf. Defaults to 0.5 .
#         beta (float): The beta parameter for the localization component of the task-aligned metric. Defaults to 6..
#                 a parameter to control the influence of box IOUs on the alignment score of an anchor box. 
#                 This is the beta parameter in equation 9 of https://arxiv.org/pdf/2108.07755.pdf.
#         epsilon (float) : small value to prevent division by zero. , defaults to 1e-9.
#         iou_type (str) : which iou type to be appled for computing overlap. Defaults to "ciou"
#         gt_bbox_format (str): the format of bounding boxes of input dataset. Defaults to 'xywh'

#     Note :
#        - with_auxiliary_dfl_regression=True , support apply df_loss in training , y_pred_bbox.shape=(b, 8400, 4+4*reg_max)
#        - with_auxiliary_dfl_regression=False , y_pred_bbox.shape=(b, 8400, 4), only use ciou_box_loss in training
    
#     Example:
#         '''Python
#         codec = YoloAnchorFreeCodec(num_classes = 4,
#                             image_shape = (640,640),
#                             max_anchor_matches=10,
#                             strides=[8, 16, 32],
#                             with_auxiliary_dfl_regression = True,
#                             alpha=0.5,
#                             beta=6.0,
#                             epsilon=1e-9,
#                             cls_label_shift = -1,
#                             gt_bbox_format = 'xywh')  
#     """
#     def __init__(
#             self,
#             num_classes,
#             image_shape : Tuple[int]=(640,640),
#             max_anchor_matches : int = 10,
#             strides: Union[Tuple[int], List[int] ] = [8, 16, 32],
#             reg_max  : int = 16,
#             with_auxiliary_dfl_regression : bool = True,
#             alpha : float = 1.0,
#             beta : float = 6.0,
#             epsilon : float= 1e-9,
#             cls_label_shift : int =  -1,
#             iou_type : str = 'ciou',
#             gt_bbox_format :str = 'xywh',
             
#         ):
#         if alpha <1.0 :
#             raise ValueError("alpha <1.0 will cause numerical problem in training progress"
#             "recommand to set as default value of 1.0"
#             )

      

#         "basic cfg"
#         self.num_classes = num_classes
#         self.cls_label_shift = cls_label_shift
#         self.max_anchor_matches = max_anchor_matches
#         self.reg_max = reg_max
#         self.with_auxiliary_dfl_regression = with_auxiliary_dfl_regression
#         self.alpha = alpha
#         self.beta = beta
#         self.epsilon = epsilon
#         self.strides = strides
#         self.image_shape = image_shape

#         '''
#         anchor_points : (8400,2)
#         anchor_srides : (8400,1)
#         '''
#         self.anchor_points, self.anchor_srides = get_anchor_points(
#             self.image_shape, self.strides
#         )
#         self.num_anchors = self.anchor_points.shape[0] #8400=80**2 +40**2 +20**2
#         print(f"anchor_points : {self.anchor_points.shape} , anchor_srides : {self.anchor_srides.shape}")
#         'init bbox format conversion'
#         if gt_bbox_format != 'xyxy':
#             self.gt_bboxes_format_transform = BBoxesFormatTransform(
#                 convert_type=gt_bbox_format+'2xyxy'
#             )
#             print(f"apply bboxes_format_transform <{gt_bbox_format+'2xyxy'}> ")

#         'init compute_ious'
#         self.compute_ious = compute_ious(
#                     mode = iou_type,
#                     boxes1_src_format ='xyxy',
#                     boxes2_src_format ='xyxy',
#                     is_aligned=False,
#                     use_masking=False
#         )


#     def get_box_metrics(self, 
#                         pred_scores, 
#                         decode_bboxes,
#                         gt_labels, 
#                         gt_bboxes, 
#                         mask_gt) ->Tuple[tf.Tensor]:
#         """
#         cls_num is number of object class, voc_cls_num=20 , coco_cls_num=80
#         pred_scores : (b,8400,cls_num)
#         decode_bboxes : (b,8400,4) , xyxy format @img_frame
#         gt_labels : (b,num_gt_bbox)
#         gt_bboxes : (b,num_gt_bbox,4), xyxy format @img_frame
#         mask_gt ( 1.or 0. float): (b,num_gt_bbox,1)
#         """
     
#         bbox_scores = tf.gather(
#                 pred_scores,
#                 tf.math.maximum(tf.cast(gt_labels, dtype=tf.int32),[0]),
#                 batch_dims=-1
#         )#(b,h*w,20)=> (b,h*w,num_gt_bbox),  tf.gather cannot support index=-1, 
#         #so if lable=-1, it will always return value of pred_scores[b,0]
#         bbox_scores = tf.transpose(
#                 bbox_scores, perm=[0, 2, 1]
#         )#(b,h*w,num_gt_bbox)=>(b,num_gt_bbox,h*w)
#         bbox_scores *= mask_gt
#         #(b,num_gt_bbox,h*w)*(b,num_gt_bbox,1) => (b,num_gt_bbox,h*w)
#         # when lable=-1,  bbox_scores get value of  pred_scores[b,0]
#         # use mask_gt to zero out these invalid values
#         overlaps =  self.compute_ious(gt_bboxes, decode_bboxes)
#         #(b,num_gt_bbox,h*w)
#         overlaps *= mask_gt
   
#         #(b,num_gt_bbox,h*w)*(b,num_gt_bbox,1) => (b,num_gt_bbox,h*w)
#         overlaps = tf.math.maximum(overlaps,0.)
#         #(b,num_gt_bbox,h*w)
#         alignment_metrics = tf.math.pow(bbox_scores,self.alpha)*tf.math.pow(overlaps,self.beta)
#         #(b,num_gt_bbox,h*w)*(b,num_gt_bbox,h*w)=>(b,num_gt_bbox,h*w)
#         # print('align_metrics', alignment_metrics)
#         # print('overlaps', overlaps)
#         return alignment_metrics, overlaps

#     def select_topk_candidates(self, alignment_metrics, mask_gt):
#         #self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
#         """

#         alignment_metrics : (b, num_gt_bbox, 8400)
#         mask_gt : (b,num_gt_bbox,1) @float [0. or 1.]
#         """
#         candidate_metrics, candidate_idxs = tf.math.top_k(
#             alignment_metrics, self.max_anchor_matches
#         ) #(b,num_gt_bbox,8400) => metrics:(b,num_gt_bbox,10),[0~1] & idxs:(b,num_gt_bbox,10) [0~8399]
#         # if all elements in axis=-1 are zeros, top_k will give invalid idx=0 and metric=0
#         # , so we need to use mask_gt to filter out these idx
   
#         candidate_idxs = tf.where(
#             tf.cast(mask_gt, dtype=tf.bool), candidate_idxs, -1
#         ) #(b,num_gt_bbox,8400) 
#         candidate_idxs = tf.where(
#             tf.greater(candidate_metrics,0.), candidate_idxs, -1
#         )# it may be removed if already overlaps.clamp(0)
#         mask_topk = tf.one_hot(
#             candidate_idxs,self.num_anchors,axis=-1
#         ) #(b,num_gt_bbox,10,num_anchors=8400) 
    
#         #tf.one_hot([-1,0,1],depth=3) will return [[0,0,0], [1,0,0], [0,1,0]]
#         mask_topk = tf.math.reduce_sum(mask_topk, axis=-2) 
#         #(b,num_gt_bbox,10,8400) => (b,num_gt_bbox,8400)
#         return mask_topk

#     def select_highest_overlaps(self,overlaps, mask_pos):
#         """
#         Args:
#         mask_pos (Tensor): shape(b, num_gt_bbox, h*w)
#         overlaps (Tensor): shape(b, num_gt_bbox, h*w)

#         Returns:
#         target_gt_idx (Tensor): shape(b, h*w)
#         fg_mask (Tensor): shape(b, h*w)
#         mask_pos (Tensor): shape(b, n_max_boxes, h*w)
#         """
#         fg_mask = tf.math.reduce_sum(mask_pos, axis=-2) 
#         # mask_pos : (b,num_gt_bbox,h*w) => fg_mask : (b,h*w) 
#         mask_multi_gts = tf.greater(fg_mask,1) 
#         # mask_multi_gts : (b,h*w) @boolean tensor
#         # to find elelment at axis=-1 is greater than 1
#         max_overlaps_idx = tf.math.argmax(overlaps, axis=1)  
#         # (b,n_max_boxes, h*w) => (b,h*w)
#         is_max_overlaps = tf.one_hot(
#             max_overlaps_idx, tf.shape(overlaps)[1]
#         ) #(b,h*w,num_gt_bbox)
#         is_max_overlaps = tf.transpose(
#             is_max_overlaps,(0,2,1)
#         ) #(b,num_gt_bbox, h*w)
#         mask_pos = tf.where(
#             mask_multi_gts[:,None,:], is_max_overlaps, mask_pos
#         )#(b,num_gt_bbox, h*w)
#         # each col in mask_pos should have one element that is 1 
#         fg_mask = tf.math.reduce_sum(
#             mask_pos, axis=-2
#         ) #(b,num_gt_bbox, h*w) => (b,h*w)
#         # target_gt_idx = tf.where(tf.cast(fg_mask, dtype=tf.bool),
#         #               tf.math.argmax(mask_pos, axis=-2),
#         #               -1)      # (b,num_gt_bbox, 8400)->(b,8400) [-1,-1,0,1,1,2]
#         '''
#         Note : finnally mask_pos should be as following
#             each col should have only one element=1 most
#             but some cols present all elemenet is zeros
#             in this condition, tf.math.argmax(mask_pos, axis=-2) will out target_gt_idx=0 
#             obviously, it's invalid, so we need to use fg_mask to filter out these invalid element
#                 anchors(h*w=6)
#                 0 1 2 3 4 5
#             gt_idx=0  [1 0 0 0 0 0]
#             gt_idx=1  [0 1 1 0 0 0]
#             gt_idx=2  [0 0 0 1 1 0]

#         '''
#         target_gt_idx = tf.math.argmax(mask_pos, axis=-2)
#         # target_gt_idx : (b, h*w)
#         #
#         #
#         return target_gt_idx, fg_mask, mask_pos
    


#     def batch_encode(self, data, y_pred):
#         r""" 
#         args :
#             data (dict[str,Tensor]) :
#                 gt_data that should contain "labels" and "bbox" keys

#                 - data["labels"] (int,[b,num_gt_bbox], [0~num_cls] ) :
#                                 gt_labels,  dtype=tf.int32
#                                 tensor value are between 0~80 @coco_dataset with padding value = 0, 
#                                 so valid labels are 1~80
#                 - data["bbox"] (float,[b,h*w,4],  [0~640.], xyxy@image_frame)  : 
#                                 gt_bbox with xywh format bbox @image_frame, dtype=self.compute_dtype
#                                 tensor value are between 0~640. @img_size=640 with padding value = 0.

#             y_pred (Tuple[Tensor]):
#                 model's prediction (pred_distri_bboxes, pred_scores)
#                 - pred_distri_bboxes (float, [b, h*w, 4],  [[0~80.],[0~40], [0,20]], xyxy@feat_frame) ) : 
#                             xyxy format @feature_frame, distance to anchor points
#                             we will transform it to pred_bboxes xyxy@feature_frame
#                 - pred_scores (float, [b, h*w, num_cls], [0~1]): 
#                             each cls score after simoid, each anchor point has num_cls
        

#         process :

#                 mask_gt (float, [b, h*w],  [0.|1.])  : 
#                 mask_pos (float, [b, num_gt_bbox, h*w],  [0.|1.]) :  

#         return :
#             data (dict[str,Tensor]) :
#                 data['y_pred'] =  (pred_bboxes, pred_scores)
#                 data['y_true'] =  (target_bboxes, target_scores)
#                 data['sample_weight'] = (box_weight, cls_weight)
        
#                 pred_bboxes (float, [b, h*w, 4], [[0~80.],[0~40], [0,20]], xyxy@feat_frame ):
#                 pred_scores (float, [b, h*w, 4], [0~1] ):
#                 target_bboxes (float, [b, h*w, 4], [[0~80.],[0~40], [0,20]], xyxy@feat_frame ):
#                 target_scores (float, [b, h*w, 4], [0.|1.]):
#                 box_weight (float, [b, h*w, 4] ):
#                 cls_weight (float, [b, h*w, 4] ):

       


#         pred_bboxes (float) : xyxy @feature frame after dist2bbox
#         gt_bboxes (float) : xywh @image frame
#         target_bboxes (float): xyxy @feature frame
#         mask_gt(float)  : (b,h*w,1) @[0. or 1.]
#         mask_pos(float) : (b,num_gt_bbox, h*w) @[0. or 1.]
#         """
    

#         pred_distri_bboxes_reg, pred_scores = y_pred
#         pred_distri_bboxes = tf.cast(pred_distri_bboxes_reg[...,:4], dtype=self.compute_dtype)
#         pred_scores = tf.cast(pred_scores, dtype=self.compute_dtype)

        
#         pred_bboxes = dist2bbox(
#             pred_distri_bboxes, self.anchor_points
#         ) #(b,8400,4) @feature_frames
       
#         gt_labels = data["labels"] + self.cls_label_shift #cls_ids
#         # gt_labels : (b,num_gt_bbox) , in coco dataset(80 class), lables is 0~80 with valid label=1~80 and padding=0 
#         # by cls_label_shift => lables becomes -1~79  with valid label=0~79 and padding=-1
#         # i.e : gt_labels with cls_label_shift=-1 [1,2,3,4,0,0] => [0,1,2,4,-1,-1] cls_label_shift=-1

#         gt_bboxes = data["bbox"] #(b,num_gt_bbox,4) xywh @image frame
#         if hasattr(self,'gt_bboxes_format_transform'):
#             gt_bboxes = self.gt_bboxes_format_transform(gt_bboxes) #(b,num_gt_bbox,4) xywh=>xyxy  @image_frames
      
#         '#1. ---------------------get mask_gt---------------------------------------------------'
#         #mask_gt = tf.reduce_all(tf.greater(gt_bboxes,0.), axis=-1, keepdims=True) #(b,num_gt_bbox,1) ?????
#         #mask_gt = tf.greater(tf.math.reduce_max(gt_bboxes, axis=-1,keepdims=True),0.) #(b,num_gt_bbox,4)=> (b,num_gt_bbox,1)
#         mask_gt = tf.cast( tf.greater_equal(gt_labels[...,None],0) , dtype=self.compute_dtype)#(b,num_gt_bbox,)=> (b,num_gt_bbox,1)

#         '#2-----------------------get pos_mask / align_metrics / overlaps  --------------------------'
        
#         mask_in_gt_boxes = is_anchor_center_within_box(
#                         self.anchor_points*self.anchor_srides, gt_bboxes
#         ) #(b,num_gt_bbox,8400) @boolean tensor
    

#         align_metrics, overlaps = self.get_box_metrics(
#                                 pred_scores = pred_scores,
#                                 decode_bboxes = pred_bboxes*self.anchor_srides,
#                                 gt_labels = gt_labels,
#                                 gt_bboxes = gt_bboxes,
#                                 mask_gt = mask_gt*mask_in_gt_boxes
#         )# align_metrics : (b,num_gt_bbox,8400), overlaps : (b,num_gt_bbox,8400) 
#         # overlaps and align_metrics both are between 0.~1.
 
#         mask_topk = self.select_topk_candidates(align_metrics, mask_gt)
#         mask_pos = mask_topk*mask_in_gt_boxes*mask_gt
        
        
#         '#3 ---------get target_gt_idx / fg_mask / mask_pos (some anchors were assigned by two labels) ----'
#         target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(overlaps, mask_pos)


#         '#4 ----- get target, bbox@image_frame----------------------------------'
#         # fg_mask : (b, h*w) @float [0. or 1.]
#         # mask_pos : (b, num_gt_bbox, h*w)  @float [0. or 1.]
        
#         target_bboxes = tf.gather(
#             gt_bboxes, target_gt_idx,batch_dims=1
#         )#(b,num_gt_bbox,4)=>(b,8400,4)
#         target_bboxes *= fg_mask[...,None]
#         #(b,8400,4)
#         # target_bbox = tf.where(
#         #     fg_mask[...,None], target_bbox, 0.
#         # )#(b,8400,4)


#         target_scores = tf.gather(
#             gt_labels, target_gt_idx, batch_dims=1
#         ) # target_scores : (b,num_gt_bbox)=>(b,8400)  [-1,num_classes-1]
#         target_scores = tf.where(
#             tf.cast(fg_mask, dtype=tf.bool), target_scores, -1
#         )#(b,8400)   ************************************
#         target_scores = tf.one_hot(
#             target_scores, self.num_classes, axis=-1
#         ) #(b,8400)=> #(b,8400,20)
      
       
#         align_metrics *= mask_pos #(b,num_gt_bbox,8400)*(b,num_gt_bbox,8400) => (b,num_gt_bbox,8400)
#         overlaps *= mask_pos
        
#         # print('pos_overlaps', overlaps)
#         # print('pos_align_metrics', align_metrics)
#         # print('target_scores', target_scores)
#         # print( "target_scores_sum : " , tf.reduce_sum(target_scores, axis=[1,2]))
#         # print("valid target_bboxes : ", tf.boolean_mask(target_bboxes, fg_mask, axis=0, name='boolean_mask'))
#         '#5 --------------- normalize targets : target_scores & target_bbox--------------'
#         pos_align_metrics  = tf.math.reduce_max(
#             align_metrics, axis=-1,keepdims=True
#         ) # (b,num_gt_bbox,8400)=>(b,num_gt_bbox,1)
#         pos_overlaps  = tf.math.reduce_max(
#             overlaps, axis=-1, keepdims=True
#         ) #(b,num_gt_bbox,8400) => (b,num_gt_bbox,1)

#         norm_align_metrics = align_metrics*pos_overlaps/(pos_align_metrics + self.epsilon) #(b,num_gt_bbox,8400)
#         norm_align_metrics = tf.math.reduce_max(
#             norm_align_metrics, axis=1
#         ) #(b,num_gt_bbox,8400)=> #(b,8400)
        
#         target_scores *= norm_align_metrics[:,:,None]
#         #bbox_labels = tf.reshape(bbox_labels, (-1, self.num_anchors, 4))

#         target_bboxes /= self.anchor_srides
#         '#5 ----------- out target, bbox@feature_frame--------------------------------------------------'
        
#         #target_scores  : (b, h*w, num_cls)
#         target_scores_sum = tf.math.maximum(tf.math.reduce_sum(target_scores), 1.) # all valid anchor points
#         box_weight = tf.math.reduce_sum(target_scores, axis=-1)*fg_mask # 應該可以刪除 (b,num_anchors)
#         box_weight = box_weight[...,None]/target_scores_sum
#         cls_weight = 1./target_scores_sum



#         if self.with_auxiliary_dfl_regression :
#             pred_distri_reg =  tf.cast(pred_distri_bboxes_reg[...,4:], dtype=self.compute_dtype)
#             if pred_distri_reg.shape[-1]!= self.reg_max*4:
#                 raise ValueError(
#                     " IF with_auxiliary_dfl_regression = True, "
#                     " codec.encoder expects number of channels in y_pred[...,4:].shape[-1] to be equal to reg_max*4"
#                     f" Received number of channels in y_pred[...,4:].shape[-1] is {pred_distri_reg.shape[-1]} "
#                     f" expected number of channels is {self.reg_max*4}"
#                 )
#             data['y_pred'] =  (tf.concat([pred_bboxes, pred_distri_reg], axis=-1) ,  pred_scores)
#         else:
#             data['y_pred'] =  (pred_bboxes ,  pred_scores)

#         data['y_true'] =  (target_bboxes, target_scores)
#         data['sample_weight'] = (box_weight, cls_weight)
#         #print(target_scores.shape, target_bboxes.shape, box_weight.shape, cls_weight.shape)
#         """
        
#         """
#         return  data
       
#     def decode_regression_to_boxes(self, pred_distri_reg_bboxes):
#         """Decodes the results of the YOLOV8Detector forward-pass into boxes.

#         Returns left / top / right / bottom predictions with respect to anchor
#         points.

#         Each coordinate is encoded with 16 predicted values. Those predictions are
#         softmaxed and multiplied by [0..15] to make predictions. The resulting
#         predictions are relative to the stride of an anchor box (and correspondingly
#         relative to the scale of the feature map from which the predictions came).
#         """

#         pred_distri_reg_bboxes = tf.keras.layers.Reshape((-1, 4, self.reg_max))(
#             pred_distri_reg_bboxes
#         )
#         pred_distri_reg_bboxes = tf.nn.softmax(
#             pred_distri_reg_bboxes, axis=-1, name='softmax'
#         )*tf.range(start=0., limit=self.reg_max, delta=1.)

#         return tf.math.reduce_sum(
#             pred_distri_reg_bboxes, axis=-1
#         )
    
#     def batch_decode(self, y_pred):
#         return NotImplemented
    





# @CODECS.register_module()
# class YoloAnchorFree:
#     VERSION = '1.0.0'
#     r""" task assignment(enocde) and prediction decode

#     """
#     def __init__(
#             self,
#             num_classes,
#             image_shape = (640,640),
#             max_anchor_matches=10,
#             strides=[8, 16, 32],
#             alpha=0.5,
#             beta=6.0,
#             epsilon=1e-9,
#             box_loss_weight=7.5,
#             class_loss_weight=0.5,
#             cls_label_shift = -1,
#             iou_type = 'ciou',
#             gt_bbox_format = 'xywh',
#         ):
#         self.num_classes = num_classes
#         self.cls_label_shift = cls_label_shift
#         self.max_anchor_matches = max_anchor_matches
#         self.alpha = alpha
#         self.beta = beta
#         self.epsilon = epsilon
#         self.strides = strides
#         self.image_shape = image_shape
#         self.box_loss_weight = box_loss_weight
#         self.class_loss_weight = class_loss_weight
        
#         '''
#         anchor_points : (8400,2)
#         anchor_srides : (8400,1)
#         '''
#         self.anchor_points, self.anchor_srides = self.get_anchors(self.image_shape, self.strides)
#         self.num_anchors = self.anchor_points.shape[0]
#         print(f"anchor_points : {self.anchor_points.shape} , anchor_srides : {self.anchor_srides.shape}")
#         #print(self.anchor_points.shape, self.anchor_srides.shape)

#         'init bbox format conversion'
#         if gt_bbox_format != 'xyxy':
#             self.gt_bboxes_format_transform = BBoxesFormatTransform(
#                 convert_type=gt_bbox_format+'2xyxy'
#             )
#             print(f"apply bboxes_format_transform <{gt_bbox_format+'2xyxy'}> ")

#         'init compute_ious'
#         self.compute_ious = compute_ious(
#                     mode = iou_type,
#                     boxes1_src_format ='xyxy',
#                     boxes2_src_format ='xyxy',
#                     is_aligned=False,
#                     use_masking=False
#         )


#     def get_anchors(
#         self,
#         image_shape,
#         strides=[8, 16, 32]
#         ):
#         all_anchors = []
#         all_strides = []
#         for stride in strides:
#             grid_coords_y = tf.cast(tf.range(0, image_shape[0],stride)+stride//2, dtype=self.compute_dtype) #(grid_h,)
#             grid_coords_x = tf.cast(tf.range(0, image_shape[1],stride)+stride//2, dtype=self.compute_dtype)
#             grid_x, grid_y = tf.meshgrid(grid_coords_x, grid_coords_y)  # (grid_h,grid_w) and (grid_h,grid_w)
#             anchor_ctr_xy = tf.stack([grid_x, grid_y], axis=-1)      # (grid_h,grid_w,2)
#             anchor_ctr_xy = tf.reshape(anchor_ctr_xy,[-1,2])
#             all_anchors.append(anchor_ctr_xy)    # [(grid_h*grid_w,2),...]
#             all_strides.append(tf.tile(tf.cast([stride], dtype=self.compute_dtype),[anchor_ctr_xy.shape[0]]))

#         all_strides = tf.concat(all_strides, axis=0)    #(8400,)
#         all_strides = tf.expand_dims(all_strides, axis=-1) #(8400,)=>(8400,1)
#         all_anchors = tf.concat(all_anchors, axis=0)    #(8400,2)
#         all_anchors = all_anchors / all_strides      #(8400,2)/(8400,1) =>(8400,2)
#         return all_anchors, all_strides

#     def dist2bbox(self, pred_distance_xyxy, anchor_points):
#         """
#         anchor_points : (8400,2) at image_frame
#         gt_bboxes : (b,num_gt_bbox,4) with xyxy format at image_frame
#         """
#         dist2lt, dist2rb = tf.split(pred_distance_xyxy, 2, axis=-1) #(b,8400,4)=>(b,8400,2)&(b,8400,2)
#         left_top_xy = anchor_points - dist2lt  #(b,8400,2)
#         right_bottom_xy = anchor_points + dist2rb #(b,8400,2)

#         return tf.concat([left_top_xy, right_bottom_xy], axis=-1)  # xyxy bbox


#     def is_anchor_center_within_box(self,anchor_points,gt_bboxes):
#         """
#         anchor_points : (8400,2) at image_frame
#         gt_bboxes : (b,num_gt_bbox,4) with xyxy format at image_frame
#         return : (b,num_gt_bbox,8400)
#         """

#         lt_cond = tf.less(gt_bboxes[...,None,:2], anchor_points)
#         rb_cond = tf.greater(gt_bboxes[...,None,2:], anchor_points)
#         mask_in_gt_boxes = tf.math.reduce_all(tf.concat([lt_cond,rb_cond],axis=-1),axis=-1)
#         return tf.cast(mask_in_gt_boxes, dtype=self.compute_dtype)

#     def get_box_metrics(self, 
#                         pred_scores, 
#                         decode_bboxes,
#                         gt_labels, 
#                         gt_bboxes, 
#                         mask_gt) ->Tuple[tf.Tensor]:
#         """
#         cls_num is number of object class, voc_cls_num=20 , coco_cls_num=80
#         pred_scores : (b,8400,cls_num)
#         decode_bboxes : (b,8400,4) , xyxy format @img_frame
#         gt_labels : (b,num_gt_bbox)
#         gt_bboxes : (b,num_gt_bbox,4), xyxy format @img_frame
#         mask_gt ( 1.or 0. float): (b,num_gt_bbox,1)
#         """
     
#         bbox_scores = tf.gather(
#                 pred_scores,
#                 tf.math.maximum(tf.cast(gt_labels, dtype=tf.int32),[0]),
#                 batch_dims=-1
#         )#(b,h*w,20)=> (b,h*w,num_gt_bbox),  tf.gather cannot support index=-1, 
#         #so if lable=-1, it will always return value of pred_scores[b,0]
#         bbox_scores = tf.transpose(
#                 bbox_scores, perm=[0, 2, 1]
#         )#(b,h*w,num_gt_bbox)=>(b,num_gt_bbox,h*w)
#         bbox_scores *= mask_gt
#         #(b,num_gt_bbox,h*w)*(b,num_gt_bbox,1) => (b,num_gt_bbox,h*w)
#         # when lable=-1,  bbox_scores get value of  pred_scores[b,0]
#         # use mask_gt to zero out these invalid values
#         overlaps =  self.compute_ious(gt_bboxes, decode_bboxes)
#         #(b,num_gt_bbox,h*w)
#         overlaps *= mask_gt
   
#         #(b,num_gt_bbox,h*w)*(b,num_gt_bbox,1) => (b,num_gt_bbox,h*w)
#         overlaps = tf.math.maximum(overlaps,0.)
#         #(b,num_gt_bbox,h*w)
#         alignment_metrics = tf.math.pow(bbox_scores,self.alpha)*tf.math.pow(overlaps,self.beta)
#         #(b,num_gt_bbox,h*w)*(b,num_gt_bbox,h*w)=>(b,num_gt_bbox,h*w)
#         # print('align_metrics', alignment_metrics)
#         # print('overlaps', overlaps)
#         return alignment_metrics, overlaps

#     def select_topk_candidates(self, alignment_metrics, mask_gt):
#         #self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
#         """

#         alignment_metrics : (b, num_gt_bbox, 8400)
#         mask_gt : (b,num_gt_bbox,1) @float [0. or 1.]
#         """
#         candidate_metrics, candidate_idxs = tf.math.top_k(
#             alignment_metrics, self.max_anchor_matches
#         ) #(b,num_gt_bbox,8400) => metrics:(b,num_gt_bbox,10),[0~1] & idxs:(b,num_gt_bbox,10) [0~8399]
#         # if all elements in axis=-1 are zeros, top_k will give invalid idx=0 and metric=0
#         # , so we need to use mask_gt to filter out these idx
   
#         candidate_idxs = tf.where(
#             tf.cast(mask_gt, dtype=tf.bool), candidate_idxs, -1
#         ) #(b,num_gt_bbox,8400) 
#         candidate_idxs = tf.where(
#             tf.greater(candidate_metrics,0.), candidate_idxs, -1
#         )# it may be removed if already overlaps.clamp(0)
#         mask_topk = tf.one_hot(
#             candidate_idxs,self.num_anchors,axis=-1
#         ) #(b,num_gt_bbox,10,num_anchors=8400) 
    
#         #tf.one_hot([-1,0,1],depth=3) will return [[0,0,0], [1,0,0], [0,1,0]]
#         mask_topk = tf.math.reduce_sum(mask_topk, axis=-2) 
#         #(b,num_gt_bbox,10,8400) => (b,num_gt_bbox,8400)
#         return mask_topk

#     def select_highest_overlaps(self,overlaps, mask_pos):
#         """
#         Args:
#         mask_pos (Tensor): shape(b, num_gt_bbox, h*w)
#         overlaps (Tensor): shape(b, num_gt_bbox, h*w)

#         Returns:
#         target_gt_idx (Tensor): shape(b, h*w)
#         fg_mask (Tensor): shape(b, h*w)
#         mask_pos (Tensor): shape(b, n_max_boxes, h*w)
#         """
#         fg_mask = tf.math.reduce_sum(mask_pos, axis=-2) 
#         # mask_pos : (b,num_gt_bbox,h*w) => fg_mask : (b,h*w) 
#         mask_multi_gts = tf.greater(fg_mask,1) 
#         # mask_multi_gts : (b,h*w) @boolean tensor
#         # to find elelment at axis=-1 is greater than 1
#         max_overlaps_idx = tf.math.argmax(overlaps, axis=1)  
#         # (b,n_max_boxes, h*w) => (b,h*w)
#         is_max_overlaps = tf.one_hot(
#             max_overlaps_idx, tf.shape(overlaps)[1]
#         ) #(b,h*w,num_gt_bbox)
#         is_max_overlaps = tf.transpose(
#             is_max_overlaps,(0,2,1)
#         ) #(b,num_gt_bbox, h*w)
#         mask_pos = tf.where(
#             mask_multi_gts[:,None,:], is_max_overlaps, mask_pos
#         )#(b,num_gt_bbox, h*w)
#         # each col in mask_pos should have one element that is 1 
#         fg_mask = tf.math.reduce_sum(
#             mask_pos, axis=-2
#         ) #(b,num_gt_bbox, h*w) => (b,h*w)
#         # target_gt_idx = tf.where(tf.cast(fg_mask, dtype=tf.bool),
#         #               tf.math.argmax(mask_pos, axis=-2),
#         #               -1)      # (b,num_gt_bbox, 8400)->(b,8400) [-1,-1,0,1,1,2]
#         '''
#         Note : finnally mask_pos should be as following
#             each col should have only one element=1 most
#             but some cols present all elemenet is zeros
#             in this condition, tf.math.argmax(mask_pos, axis=-2) will out target_gt_idx=0 
#             obviously, it's invalid, so we need to use fg_mask to filter out these invalid element
#                 anchors(h*w=6)
#                 0 1 2 3 4 5
#             gt_idx=0  [1 0 0 0 0 0]
#             gt_idx=1  [0 1 1 0 0 0]
#             gt_idx=2  [0 0 0 1 1 0]

#         '''
#         target_gt_idx = tf.math.argmax(mask_pos, axis=-2)
#         # target_gt_idx : (b, h*w)
#         #
#         #
#         return target_gt_idx, fg_mask, mask_pos
    
   

#     def batch_encode(self, data, y_pred):
#         r""" 
#         args :
#             data (dict[str,Tensor]) :
#                 gt_data that should contain "labels" and "bbox" keys

#                 - data["labels"] (int,[b,num_gt_bbox], [0~num_cls] ) :
#                                 gt_labels,  dtype=tf.int32
#                                 tensor value are between 0~80 @coco_dataset with padding value = 0, 
#                                 so valid labels are 1~80
#                 - data["bbox"] (float,[b,h*w,4],  [0~640.], xyxy@image_frame)  : 
#                                 gt_bbox with xywh format bbox @image_frame, dtype=self.compute_dtype
#                                 tensor value are between 0~640. @img_size=640 with padding value = 0.

#             y_pred (Tuple[Tensor]):
#                 model's prediction (pred_distri_bboxes, pred_scores)
#                 - pred_distri_bboxes (float, [b, h*w, 4],  [[0~80.],[0~40], [0,20]], xyxy@feat_frame) ) : 
#                             xyxy format @feature_frame, distance to anchor points
#                             we will transform it to pred_bboxes xyxy@feature_frame
#                 - pred_scores (float, [b, h*w, num_cls], [0~1]): 
#                             each cls score after simoid, each anchor point has num_cls
        

#         process :

#                 mask_gt (float, [b, h*w],  [0.|1.])  : 
#                 mask_pos (float, [b, num_gt_bbox, h*w],  [0.|1.]) :  

#         return :
#             data (dict[str,Tensor]) :
#                 data['y_pred'] =  (pred_bboxes, pred_scores)
#                 data['y_true'] =  (target_bboxes, target_scores)
#                 data['sample_weight'] = (box_weight, cls_weight)
        
#                 pred_bboxes (float, [b, h*w, 4], [[0~80.],[0~40], [0,20]], xyxy@feat_frame ):
#                 pred_scores (float, [b, h*w, 4], [0~1] ):
#                 target_bboxes (float, [b, h*w, 4], [[0~80.],[0~40], [0,20]], xyxy@feat_frame ):
#                 target_scores (float, [b, h*w, 4], [0.|1.]):
#                 box_weight (float, [b, h*w, 4] ):
#                 cls_weight (float, [b, h*w, 4] ):

       


#         pred_bboxes (float) : xyxy @feature frame after dist2bbox
#         gt_bboxes (float) : xywh @image frame
#         target_bboxes (float): xyxy @feature frame
#         mask_gt(float)  : (b,h*w,1) @[0. or 1.]
#         mask_pos(float) : (b,num_gt_bbox, h*w) @[0. or 1.]
#         """
    

#         pred_distri_bboxes, pred_scores = y_pred

#         pred_bboxes = self.dist2bbox(pred_distri_bboxes, self.anchor_points) #(b,8400,4) @feature_frames
       
#         gt_labels = data["labels"] + self.cls_label_shift #cls_ids
#         # gt_labels : (b,num_gt_bbox) , in coco dataset(80 class), lables is 0~80 with valid label=1~80 and padding=0 
#         # by cls_label_shift => lables becomes -1~79  with valid label=0~79 and padding=-1
#         # i.e : gt_labels with cls_label_shift=-1 [1,2,3,4,0,0] => [0,1,2,4,-1,-1] cls_label_shift=-1

#         gt_bboxes = data["bbox"] #(b,num_gt_bbox,4) xywh @image frame
#         if hasattr(self,'gt_bboxes_format_transform'):
#             gt_bboxes = self.gt_bboxes_format_transform(gt_bboxes) #(b,num_gt_bbox,4) xywh=>xyxy  @image_frames
      
#         '#1. ---------------------get mask_gt---------------------------------------------------'
#         #mask_gt = tf.reduce_all(tf.greater(gt_bboxes,0.), axis=-1, keepdims=True) #(b,num_gt_bbox,1) ?????
#         #mask_gt = tf.greater(tf.math.reduce_max(gt_bboxes, axis=-1,keepdims=True),0.) #(b,num_gt_bbox,4)=> (b,num_gt_bbox,1)
#         mask_gt = tf.cast( tf.greater_equal(gt_labels[...,None],0) , dtype=self.compute_dtype)#(b,num_gt_bbox,)=> (b,num_gt_bbox,1)
#         #mask_gt = ops.all(y["boxes"] > -1.0, axis=-1, keepdims=True)
#         #print(mask_gt)
#         '#2-----------------------get pos_mask / align_metrics / overlaps  --------------------------'
        
#         mask_in_gt_boxes = self.is_anchor_center_within_box(
#                         self.anchor_points*self.anchor_srides, gt_bboxes
#         ) #(b,num_gt_bbox,8400) @boolean tensor
    

#         align_metrics, overlaps = self.get_box_metrics(
#                                 pred_scores = pred_scores,
#                                 decode_bboxes = pred_bboxes*self.anchor_srides,
#                                 gt_labels = gt_labels,
#                                 gt_bboxes = gt_bboxes,
#                                 mask_gt = mask_gt
#         )# align_metrics : (b,num_gt_bbox,8400), overlaps : (b,num_gt_bbox,8400) 
#         # overlaps and align_metrics both are between 0.~1.
 
#         mask_topk = self.select_topk_candidates(align_metrics, mask_gt)
#         mask_pos = mask_topk*mask_in_gt_boxes*mask_gt
        
        
#         '#3 ---------get target_gt_idx / fg_mask / mask_pos (some anchors were assigned by two labels) ----'
#         target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(overlaps, mask_pos)


#         '#4 ----- get target, bbox@image_frame----------------------------------'
#         # fg_mask : (b, h*w) @float [0. or 1.]
#         # mask_pos : (b, num_gt_bbox, h*w)  @float [0. or 1.]
        
#         target_bboxes = tf.gather(
#             gt_bboxes, target_gt_idx,batch_dims=1
#         )#(b,num_gt_bbox,4)=>(b,8400,4)
#         target_bboxes *= fg_mask[...,None]
#         #(b,8400,4)
#         # target_bbox = tf.where(
#         #     fg_mask[...,None], target_bbox, 0.
#         # )#(b,8400,4)


#         target_scores = tf.gather(
#             gt_labels, target_gt_idx, batch_dims=1
#         ) # target_scores : (b,num_gt_bbox)=>(b,8400)  [-1,num_classes-1]
#         target_scores = tf.where(
#             tf.cast(fg_mask, dtype=tf.bool), target_scores, -1
#         )#(b,8400)   ************************************
#         target_scores = tf.one_hot(
#             target_scores, self.num_classes, axis=-1
#         ) #(b,8400)=> #(b,8400,20)
      
       
#         align_metrics *= mask_pos #(b,num_gt_bbox,8400)*(b,num_gt_bbox,8400) => (b,num_gt_bbox,8400)
#         overlaps *= mask_pos
        
#         # print('pos_overlaps', overlaps)
#         # print('pos_align_metrics', align_metrics)
#         # print('target_scores', target_scores)
#         # print( "target_scores_sum : " , tf.reduce_sum(target_scores, axis=[1,2]))
#         # print("valid target_bboxes : ", tf.boolean_mask(target_bboxes, fg_mask, axis=0, name='boolean_mask'))
#         '#5 --------------- normalize targets : target_scores & target_bbox--------------'
#         pos_align_metrics  = tf.math.reduce_max(
#             align_metrics, axis=-1,keepdims=True
#         ) # (b,num_gt_bbox,8400)=>(b,num_gt_bbox,1)
#         pos_overlaps  = tf.math.reduce_max(
#             overlaps, axis=-1, keepdims=True
#         ) #(b,num_gt_bbox,8400) => (b,num_gt_bbox,1)

#         norm_align_metrics = align_metrics*pos_overlaps/(pos_align_metrics + self.epsilon) #(b,num_gt_bbox,8400)
#         norm_align_metrics = tf.math.reduce_max(
#             norm_align_metrics, axis=1
#         ) #(b,num_gt_bbox,8400)=> #(b,8400)
        
#         target_scores *= norm_align_metrics[:,:,None]
#         #bbox_labels = tf.reshape(bbox_labels, (-1, self.num_anchors, 4))

#         target_bboxes /= self.anchor_srides
#         '#5 ----------- out target, bbox@feature_frame--------------------------------------------------'
        
#         #target_scores  : (b, h*w, num_cls)
#         target_scores_sum = tf.math.maximum(tf.math.reduce_sum(target_scores), 1.) # all valid anchor points
#         box_weight = tf.math.reduce_sum(target_scores, axis=-1)*fg_mask # 應該可以刪除
#         box_weight = self.box_loss_weight*box_weight/target_scores_sum

#         cls_weight = self.class_loss_weight/target_scores_sum

#         data['y_pred'] =  (pred_bboxes, pred_scores)
#         data['y_true'] =  (target_bboxes, target_scores)
#         data['sample_weight'] = (box_weight, cls_weight)

#         #print(target_scores.shape, target_bboxes.shape, box_weight.shape, cls_weight.shape)
#         #
#         # y_true = (target_bboxes, target_scores)
#         # sample_weight = (box_weight, cls_weight)
#         """
#         """
#         return  data
    
#     def batch_decode(self, y_pred):
#         return NotImplemented