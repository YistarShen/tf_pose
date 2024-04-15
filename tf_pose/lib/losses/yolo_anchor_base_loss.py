import tensorflow as tf
from tensorflow import Tensor
import math
from lib.Registers import LOSSES

@LOSSES.register_module()
class YoloAnchorBaseLoss(tf.losses.Loss):
    '''
    YOLO_FPN_ANCHORS_CFG = {
        "img_size": [640,640], 
        "anchor_sizes": [ [[12,16],[19,36],[40,28]], [[36,75], [76,55], [72,146]], [[142,110], [192,243],[459,401]] ],
        "strides" : [8, 16, 32],
         "feature_map_shapes": [80,40,20],
        "fpn_balance" : [4., 1., 0.4],
    }
    '''

    def __init__(self, 
            fpn_feat_size,
            fpn_anchor_size_list,
            fpn_balance_factor,
            model_input_size = (640,640), 
            num_classes=4,
            alpha=0.25, 
            gamma=2.0) :
        super(YoloAnchorBaseLoss, self).__init__(reduction="auto", name="yolo_loss")
        'hyper_params config'

        self._input_size = model_input_size # _input_size = (640,640)
        self._feat_shapes = fpn_feat_size# 80, 40, 20, (sqaure shape)
        self._num_classes = num_classes
        self._num_anchors = len(fpn_anchor_size_list)
        self._anchor_size_3x2 = tf.constant(fpn_anchor_size_list, dtype=tf.float32) #(3,2)
        self.eps = 1e-7
        self.pi_square = tf.cast(math.pi**2, dtype=tf.float32)

        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = 0.01
        self.fpn_balance = fpn_balance_factor #tf.constant([4.0, 1.0, 0.4],dtype=tf.float32) # P3-P5
        self.box_ratio = 0.05 
        self.cls_ratio = 0.5 #need to modify????
        self.obj_ratio = 1. 

        def gen_grid_xy(feat_shape):
            grid_coords_y = tf.cast(tf.range(0, feat_shape), dtype=tf.float32)
            grid_coords_x = tf.cast(tf.range(0, feat_shape), dtype=tf.float32)
            grid_x, grid_y = tf.meshgrid(grid_coords_x, grid_coords_y)  # (grid_coords_x,grid_height)  
            grid_xy = tf.concat([grid_x[...,None],grid_y[...,None]], axis=-1) # (grid_coords_x,grid_height, 2) 
            print(f'grid_xy : {grid_xy.shape}')
      
            return grid_xy

        self.grid_xy = gen_grid_xy(self._feat_shapes)

    def _generate_ciou_map(self, boxes_pred, boxes_true):
        """
        boxes_pred : (b, feat_h, feat_w, num_anchors=3, 4), here, box type is center_xywh
        boxes_true : (b, feat_h, feat_w, num_anchors=3, 4), here, box type is center_xywh
        """

        boxes_pred_mins =  boxes_pred[...,:2] - boxes_pred[...,2:4]/2.
        boxes_pred_maxes =  boxes_pred[...,:2] + boxes_pred[...,2:4]/2.
        #(b, feat_h, feat_w, num_anchors=3, 4 ), here, box type is conner_xyxy
        boxes_pred_xyxy = tf.concat([boxes_pred[..., :2]-boxes_pred[..., 2:]/2., boxes_pred[...,:2] + boxes_pred[...,2:]/2.],axis=-1) #  (xy_min,xy_max)
        boxes_true_xyxy = tf.concat([boxes_true[..., :2]-boxes_true[..., 2:]/2., boxes_true[...,:2] + boxes_true[...,2:]/2.],axis=-1) #  (xy_min,xy_max)

        ld = tf.maximum(boxes_pred_xyxy[...,:2], boxes_true_xyxy[...,:2])   #(b,feat_h,feat_w,3, 2)
        ru = tf.minimum(boxes_pred_xyxy[...,2:], boxes_true_xyxy[...,2:])   #(b,feat_h,feat_w,3, 2) 

        intersection = tf.maximum(0.0, ru - ld)                #(b,feat_h,feat_w,3, 2)
        intersection_area = intersection[..., 0] * intersection[..., 1]   #(b,feat_h,feat_w,3)

        boxes_pred_area = boxes_pred[...,2]*boxes_pred[...,3]          #(b,feat_h,feat_w,3)
        boxes_true_area = boxes_true[...,2]*boxes_true[...,3]          #(b,feat_h,feat_w,3)

        union_area = tf.maximum(boxes_pred_area + boxes_true_area - intersection_area, 1e-8)  #(b,feat_h,feat_w,3)
        #iou = tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)             #(b,feat_h,feat_w,3)
        iou = intersection_area / union_area
        'ciou'
        ctr_distance = tf.reduce_sum( tf.math.square(boxes_pred[...,:2]-boxes_true[...,:2]) , axis=-1) #(b,feat_h,feat_w,3,2) => (b,feat_h,feat_w,3)
        enclose_mins = tf.minimum(boxes_pred_xyxy[...,:2], boxes_true_xyxy[...,:2])        #(b,feat_h,feat_w,3,2)
        enclose_maxes = tf.maximum(boxes_pred_xyxy[...,2:], boxes_true_xyxy[...,2:])       #(b,feat_h,feat_w,3,2)
        enclose_wh = tf.maximum(enclose_maxes-enclose_mins, 0.)                   #(b,feat_h,feat_w,3,2)

        enclose_diagonal = tf.reduce_sum( tf.math.square(enclose_wh) , axis=-1)          #(b,feat_h,feat_w,3)
        ciou = iou - 1.0 * ctr_distance/ tf.maximum(enclose_diagonal, 1e-8)           #(b,feat_h,feat_w,3)

        arctan_wh_true = tf.math.atan2(boxes_true[...,2], tf.maximum(boxes_true[...,3], self.eps))  #(b,feat_h,feat_w,3)
        arctan_wh_pred = tf.math.atan2(boxes_pred[...,2], tf.maximum(boxes_pred[...,3], self.eps))  #(b,feat_h,feat_w,3)

        v = 4.*tf.pow( arctan_wh_true - arctan_wh_pred,2)/self.pi_square #(b,feat_h,feat_w,3)
        alpha = v /(v-iou+1.+self.eps)
        ciou = ciou - alpha * v #(b,feat_h,feat_w,3)

        return tf.clip_by_value(ciou, 0.0, 1.0)

    def call(self, y_true, y_pred):
        """
        calculate Loss for per output features
        y_pred: (batch_size, 80,80,3, 4+num_cls+1)/ (batch_size, 40,40,3, 6) /(batch_size, 20,20,3, 6) from Model output
        y_true: (batch_size, 80,80,3, 4+1+1) / (batch_size, 40,40,3, 6) / (batch_size, 20,20,3, 6) from targets generator, 
            normalized true bbox_ctr_xywh (not nomalized)

        #sample_weights : (batch_size, num_gt, 4+1)

        #Note : Here, 4+1+1 means [ bbox_xywh, cls, Object_Mask]
        """
        'y_true'
        box_true = y_true[...,:4]     #(b,feat_h,feat_w,3,4) normalize
        cls_true = y_true[...,4]     #(b,feat_h,feat_w,3)
        obj_mask = y_true[...,-1]    #(b,feat_h,feat_w,3)
        normalizer = tf.maximum(tf.reduce_sum(obj_mask, axis=[1,2,3]), 1.) #(b,) num_positive_samples
        'y_pred'
        reg_pred = y_pred[...,:4]            #(b,feat_h,feat_w,3,4)
        cls_pred = y_pred[...,4:4+self._num_classes]  #(b,feat_h,feat_w,3,mum_cls)
        obj_pred = y_pred[...,-1]            #(b,feat_h,feat_w,3)

        'decode predictions'
        box_xy = (tf.nn.sigmoid(reg_pred[..., :2])*2-0.5 + self.grid_xy[None,:,:,None,:])/tf.cast(self._feat_shapes, dtype=tf.float32) #(b,feat_h,feat_w,3,2)
        box_wh = ( (tf.nn.sigmoid(reg_pred[..., 2:4])*2)**2 )*self._anchor_size_3x2[None, None, None,:,:]/tf.cast(self._input_size[0], dtype=tf.float32) #(b,feat_h, feat_w, 3, 2)
        box_pred = tf.concat([box_xy,box_wh], axis=-1)        #(b,feat_h,feat_w,3, 2)
        #-----------------------------------------------------------#
        #   reg_loss
        #-----------------------------------------------------------#
        'calculate ciou'
        ciou = self._generate_ciou_map(box_pred, box_true)           #(b,feat_h,feat_w,3)
        ciou_loss = tf.reduce_sum(obj_mask * (1. - ciou), axis=[1,2,3])  #(b,)
        reg_loss = tf.math.divide_no_nan(ciou_loss, normalizer)*self.box_ratio  

        #-----------------------------------------------------------#
        #   class_loss (Focal Loss)
        #-----------------------------------------------------------#
        cls_label = tf.one_hot(tf.cast(cls_true, dtype=tf.int32), depth=self._num_classes, dtype=tf.float32 )   #(b,feat_h,_feat_w,3)=>(b,feat_h,feat_w,3,mum_cls)
        cls_true = cls_label * (1.0 - self.label_smoothing) + self.label_smoothing/self._num_classes       #(b,feat_h,_feat_w,3)=>(b,feat_h,feat_w,3,mum_cls)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=cls_true, logits=cls_pred)           #(b,feat_h,feat_w,3,mum_cls)

        if self.gamma :
            pred_probs = tf.nn.sigmoid(cls_pred)                             #(b,feat_h,feat_w,3,mum_cls)
            pt = tf.where(tf.equal(cls_label, 1.0), pred_probs, 1 - pred_probs)          #(b,feat_h,feat_w,3,mum_cls)
            alpha = tf.where(tf.equal(cls_label, 1.0), self.alpha, (1.0 - self.alpha))      #(b,feat_h,feat_w,3,mum_cls)
            cls_loss = alpha * tf.pow(1.0 - pt, self.gamma)*cross_entropy  #(b,feat_h,feat_w,3,mum_cls) 
        else:
            cls_loss = cross_entropy                        #(b,feat_h,feat_w,3,mum_cls) 

        cls_loss = tf.math.divide_no_nan(tf.reduce_sum(obj_mask[...,None]*cls_loss, axis=[1,2,3,4]), normalizer)*self.cls_ratio   ##(b,feat_h,feat_w,3,mum_cls) => (b,)
        cls_loss /= self._num_classes

        #-----------------------------------------------------------#
        #   object loss
        #-----------------------------------------------------------#  
        obj_true = obj_mask*ciou  #(b,feat_h,feat_w,3)
        obj_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_true, logits=obj_pred)      #(b,feat_h,feat_w,3)
        obj_loss = tf.reduce_mean(obj_loss, axis=[1,2,3])*self.fpn_balance*self.obj_ratio        #(b,feat_h,feat_w,3)=>(b)

        loss = reg_loss + cls_loss + obj_loss
        return loss 