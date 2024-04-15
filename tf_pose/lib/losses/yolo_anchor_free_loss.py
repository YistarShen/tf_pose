import tensorflow as tf
from lib.Registers import LOSSES
from lib.codecs.bounding_box import bbox2dist, get_anchor_points, compute_ious
from typing import Optional
import numpy as np

#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
@LOSSES.register_module()
class YoloAnchorFreeBBoxLoss(tf.losses.Loss):
    VERSION = "1.0.0"
    R"""YoloAnchorFreeBBoxLoss (ciou +dfl)

    
    #https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/yolov8/losses.py#L329
    Valid Reduction Keys are "('auto', 'none', 'sum', 'sum_over_batch_size')"
    """
    def __init__(self, 
                iou_type : str='ciou',
                pred_boxes_format = 'xyxy',
                true_boxes_format = 'xyxy',
                to_multiply_batch_size  : bool = False,
                bbox_weight : float = 7.5,
                dfl_weight = 1.5,
                use_dfl : bool = False,
                reg_max : int = 16,
                image_shape : int=(640,640),
                strides : list =[8, 16, 32],
                **kwargs) :
        super().__init__(**kwargs)
        if bbox_weight<=1. :
            raise ValueError("bbox_weight must be greater than 1.,  "
                    f"but got {bbox_weight} @{self.__class__.__name__}"
            )
        self.bbox_weight = bbox_weight
        self.use_dfl = use_dfl
        self.to_multiply_batch_size = to_multiply_batch_size
        'init compute_ious'
        self.compute_cious = compute_ious(
                mode = iou_type,
                boxes1_src_format = pred_boxes_format,
                boxes2_src_format = true_boxes_format,
                is_aligned=True,
                use_masking=False
        )
        if self.use_dfl :
            if dfl_weight<=0. :
                raise ValueError("dfl_weight must be greater than 0., if use_dfl"
                        f"but got {dfl_weight} @{self.__class__.__name__}"
                )
            self.dfl_weight = dfl_weight
            self.anchor_points, _ = get_anchor_points(
                image_shape, strides=strides
            ) #(b, 8400,2)
            self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, reduction='none'
            )
            self.reg_max = reg_max
                
    def distribution_focal_loss (
            self, target_dist, pred_dist,                         
        ):
        """ dfl
        target_dist : (b, 8400, 4) <float>
        pred_dist : (b, 8400, 64)  <float>
        """
        if pred_dist.shape[1]!=target_dist.shape[1]:
            raise ValueError(
                "distribution_focal_loss expects pred_dist.shape[1] to be equal to target_dist.shape[1]"
                f"pred_dist={pred_dist.shape} and  y_pred={target_dist.shape}."
            )
        pred_dist = tf.reshape(pred_dist,[-1, target_dist.shape[1], pred_dist.shape[-1]//self.reg_max, self.reg_max])
        tl_int = tf.cast(target_dist, dtype=tf.int32) #@int
        tr_int = tl_int + 1  # target right @int
        wl_fp = tf.cast(tr_int, dtype=tf.float32) - target_dist  # weight left @float
        wr_fp = 1. - wl_fp  # weight right'@float
        
        dfl = self.cross_entropy_loss(
            tf.one_hot(tl_int, self.reg_max), pred_dist
        )*wl_fp #dfl (b, 8400, 4) => 
        dfl += self.cross_entropy_loss(
            tf.one_hot(tr_int, self.reg_max), pred_dist
        )*wr_fp
        #dfl (b, 8400, 4) => (b, 8400)
        return tf.math.reduce_mean(dfl, axis=-1)
    
    def distribution_focal_loss (
            self, target_dist, pred_distri,                         
        ):
        """ dfl
        target_dist : (b, 8400, 4) <float>
        pred_dist : (b, 8400, 64)  <float>
        """
        if pred_distri.shape[1]!=target_dist.shape[1]:
            raise ValueError(
                "distribution_focal_loss expects pred_distri.shape[1] to be equal to target_dist.shape[1]"
                f"pred_distri={pred_distri.shape} and  y_pred={target_dist.shape}."
            )
        pred_dist_reg = tf.reshape(
            pred_distri,
            [-1, target_dist.shape[1], pred_distri.shape[-1]//self.reg_max, self.reg_max]
        )#(b, 8400, 4, reg_max)
        if pred_dist_reg.shape[-2]!=target_dist.shape[-1]:
            raise ValueError(
                "distribution_focal_loss expects pred_dist_reg.shape[-2] to be equal to target_dist.shape[-1]"
                f"but recieve pred_dist_reg={pred_dist_reg.shape} and  y_pred={target_dist.shape}."
            )
        tl_int = tf.cast(target_dist, dtype=tf.int32) #@int
        tr_int = tl_int + 1  # target right @int
        wl_fp = tf.cast(tr_int, dtype=tf.float32) - target_dist  # weight left @float
        wr_fp = 1. - wl_fp  # weight right'@float
        dfl = self.cross_entropy_loss(
            tf.one_hot(tl_int, self.reg_max), pred_dist_reg
        )*wl_fp #dfl (b, 8400, 4) => 
        dfl += self.cross_entropy_loss(
            tf.one_hot(tr_int, self.reg_max), pred_dist_reg
        )*wr_fp
        #dfl (b, 8400, 4) => (b, 8400)
        return tf.math.reduce_mean(dfl, axis=-1)
    

    def call(self,y_true, y_pred):
        """ 
        y_true : (b, 8400, 4) , bbox_xyxy format
        y_pred : (b, 8400, 4+reg_max*4)
        sample_weight : (b, 8400)
        """
        pred_bboxes = y_pred[..., :4]
        if y_true.shape[-2] != y_pred.shape[-2]:
            raise ValueError(
                "CIoULoss expects number of boxes in y_pred to be equal to the "
                "number of boxes in y_true. Received number of boxes in "
                f"y_true={y_true.shape[-2]} and number of boxes in "
                f"y_pred={y_pred.shape[-2]}."
            )
        ciou = self.compute_cious(y_true, pred_bboxes)  #(b,8400,4) and (b,8400,4) => (b, 8400)
        ciou_loss = (1.-ciou)*self.bbox_weight #(b, 8400)
        if self.use_dfl:
            pred_distri_bboxes = y_pred[..., 4:] #(b, 8400, 64)
            if pred_distri_bboxes.shape[-1]!=4*self.reg_max:
                raise ValueError(
                    f"df_Loss expects pred_distri_bboxes.shape[-1] to be {4*self.reg_max}. "
                    f" Received pred_distri_bboxes.shape[-1]={pred_distri_bboxes.shape[-1]}."
                )
            target_dist_ltrb = bbox2dist(
                y_true, self.anchor_points, self.reg_max
            )#(b,8400,4) -> (b,8400,4)
            df_loss = self.distribution_focal_loss(
                target_dist_ltrb, pred_distri_bboxes
            )*self.dfl_weight #(b, 8400)
        # df_loss : (b,8400), ciou_loss : (b,8400)
        loss = ciou_loss + df_loss if  self.use_dfl else ciou_loss

        return loss*pred_bboxes.shape[0] if self.to_multiply_batch_size else loss
    
#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
@LOSSES.register_module()
class YoloAnchorFreeClassLoss(tf.losses.Loss):
    VERSION = "1.0.0"
    R"""YoloAnchorFreeClassLoss (ciou +dfl)

    tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction=losses_utils.ReductionV2.AUTO,
        name='binary_crossentropy'
    )


    """
    def __init__(self, 
                class_loss_weight : float = 0.5,
                to_multiply_batch_size  : bool = False,
                **kwargs) :
        super().__init__(**kwargs)
        if class_loss_weight<=0. :
            raise ValueError("class_loss_weight must be greater than 0.,  "
                    f"but got {class_loss_weight} @{self.__class__.__name__}"
            )
        self.class_loss_weight = class_loss_weight
        self.to_multiply_batch_size = to_multiply_batch_size
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction='none')

    def call(self,y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError(
                "YoloAnchorFreeClassLoss expects  y_pred.shape to be equal to y_true.shape"
                f"but got y_pred.shape={y_pred.shape} and y_true.shape={y_true.shape}"
            )
        loss =self.cross_entropy(y_true, y_pred)*self.class_loss_weight   #(b, 8400,num_cls) -#(b, 8400)
        #loss = tf.math.reduce_sum(self.cross_entropy(y_true, y_pred), axis=-1)*sample_weight 
        return loss*y_pred.shape[0] if self.to_multiply_batch_size else loss
    

#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
@LOSSES.register_module()
class YoloAnchorFreePoseLoss(tf.losses.Loss):
    VERSION = "1.0.0"
    R"""YoloAnchorFreeClassLoss (ciou +dfl)

    tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction=losses_utils.ReductionV2.AUTO,
        name='binary_crossentropy'
    )

    """
    OKS_SIGMA = (
        np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])/ 10.0
    )
    def __init__(
            self, 
            oks_sigmas : Optional[float] = None,
            kobj_weight : float = 1.0,
            pose_weight : float = 12.0,
            num_joints : int = 17,
            **kwargs
        ) :
        super().__init__(**kwargs)

        if kobj_weight<=0. or pose_weight<=0. :
            raise ValueError(
                f"kobj_weight and pose_weight must be both greater than 0. @{self.__class__.__name__} "
                f"but got kobj_weight : {kobj_weight} and pose_weight:{pose_weight}"
            )
        
        self.kobj_weight = kobj_weight
        self.pose_weight = pose_weight
        if oks_sigmas is None :
            oks_sigmas = tf.ones(
                shape=(num_joints,),dtype=tf.float32
            )/float(num_joints)
        # #(2 * oks_sigmas)**2
        self.sigmas = tf.math.square(
            2.0*tf.constant(oks_sigmas, dtype=tf.float32)
        )
        self.bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,reduction='none'
        )

    def call(self,y_true, y_pred):
        r'''
        y_true : (b,num_anchors,17,3),   (x,y,area if visable else 0.)
        y_pred : (b, num_anchors,17,3)
        return  : (b,num_anchors,) 
        '''
        if y_true.shape[-1] != 3:
            raise ValueError(
                "YoloAnchorFreePoseLoss expects y_true.shape to  be like (b,num_anchors,17,3)"
                "here the last dims means (kpt_x,kpt_y, area of gt-bbox of object) "
                f"but got y_true.shape={y_true.shape}"
            )
        

        # if self.sigmas.shape[0]!=y_true.shape[-2]:
        #     raise ValueError(
        #         "YoloAnchorFreePoseLoss expects y_true.shape to  be like (b,num_anchors,17,3)"
        #         "here the last dims means (kpt_x,kpt_y, area of gt-bbox of object) "
        #         f"but got y_true.shape={y_true.shape}"
        #     )         
        kpts_loss = 0.
        kpts_obj_loss = 0.
        area = tf.reduce_max(y_true[...,-1], axis=-1)  #(b,num_anchors,17)-> #(b,num_anchors,)
        kpt_mask = tf.cast( 
            tf.greater( y_true[...,-1], 0.), dtype=tf.float32
        )#(b,num_anchors,17)
        kpt_loss_factor = tf.cast(
            tf.shape(kpt_mask)[-1], dtype=tf.float32
        )/(tf.reduce_sum(kpt_mask, axis=-1) + 1e-9) #(b,num_anchors,17)>  #(b,num_anchors) @[1~17]

        d = tf.reduce_sum(
            tf.math.square(y_true[...,:2]-y_pred[...,:2]),
            axis=-1               
        ) #(b,num_anchors,17,2) -> #(b,num_anchors,17)
        e = d /self.sigmas/(area[...,None] + 1e-9)/2.  #(b,num_anchors,17)
        kpts_loss = (
            1 - tf.math.exp(-e)
        )#(b,num_anchors,17)
        kpts_loss =  tf.reduce_mean(
            kpt_loss_factor[..., None]*kpts_loss*kpt_mask , axis=-1
        ) #(b,num_anchors,17) => (b,num_anchors,17) 

        if  tf.shape(y_pred)[-1] == 3 :
            kpts_obj_loss = self.bce(
                kpt_mask, y_pred[...,2]
            ) #(b,num_anchors,17)-> #(b,num_anchors,)

        ''' 
            kobj_loss : (b,num_anchors,)
            kpts_loss : (b,num_anchors,)
            return  : (b,num_anchors,) 
        '''
        return kpts_obj_loss*self.kobj_weight + kpts_loss*self.pose_weight    