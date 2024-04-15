import tensorflow as tf
from tensorflow import keras
import math
from lib.datasets.transforms import BBoxesFormatTransform

def compute_area(box):
    """Computes area for bounding boxes

    Args:
        box: [N, 4] or [batch_size, N, 4] float Tensor, either batched
        or unbatched boxes.
    Returns:
        a float Tensor of [N] or [batch_size, N]
    """
    return tf.math.reduce_prod(box[...,2:]-box[...,:2], axis=-1)


def compute_intersection(boxes1, boxes2, is_aligned=True):
    """Computes intersection area between two sets of boxes.

    Args:
      boxes1: [N, 4] or [batch_size, N, 4] float Tensor boxes.
      boxes2: [M, 4] or [batch_size, M, 4] float Tensor boxes.
    Returns:
      a [N, M] or [batch_size, N, M] float Tensor.
    """

    # print(tf.shape(boxes2), boxes1.shape)
    # if tf.shape(boxes1)[:]!=tf.shape(boxes2)[:]:
    #   raise ValueError(
    #         "if is_aligned = True , boxes1.shape must be equal toboxes2.shape"
    #         f"Received boxes1.shape={tf.shape(boxes1)} and boxes2.shape={tf.shape(boxes2)}"
    #     )

    if is_aligned :
        yx_min = tf.maximum(boxes1[...,:2], boxes2[...,:2])  #maximum((b,m,2),(b,m,2)) => (b,m,2)
        yx_max = tf.minimum(boxes1[...,2:], boxes2[...,2:])  #maximum((b,m,2),(b,m,2)) => (b,m,2)
    else:
        yx_min = tf.maximum(boxes1[...,None,:2], boxes2[...,None,:,:2]) #maximum((b,m,2),(b,n,2)) => (b,m,n,2)
        yx_max = tf.minimum(boxes1[...,None,2:], boxes2[...,None,:,2:])  #maximum((b,m,2),(b,n,2)) => (b,m,n,2)

    intersection = tf.maximum(tf.cast(0.,dtype=yx_max.dtype), yx_max - yx_min)                #(b,m,n,2) / (b,m,2)
    intersection_area = intersection[..., 0] * intersection[..., 1]      #(b,m,n) / (b,m)
    return intersection_area


class compute_ious:
    VERSION = '1.0.0'
    r""" compute_ious (iou,ciou,giou,siou)


    #https://github.com/open-mmlab/mmdetection/blob/main/mmdet/structures/bbox/bbox_overlaps.py
    #https://github.com/open-mmlab/mmyolo/blob/8c4d9dc503dc8e327bec8147e8dc97124052f693/mmyolo/models/losses/iou_loss.py
    #https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/iou.py



        boxes1 = tf.constant([[20.,30., 60., 90.],[10.,15., 50., 60.]])
        boxes2 = tf.constant([[25.,35., 75., 80.], [15.,35., 100., 80.]])
        print(tf.shape(boxes2), boxes1.shape)
        res = compute_ious(mode = 'ciou',is_aligned=False)(boxes1,boxes2)
        print(res.shape)

    """
    valid_boxes_src_format  = ["cxcywh","yxyx","xyxy","xywh"]

    def __init__(self,
            mode = 'iou',
            is_aligned=True,
            use_masking=False,
            boxes1_src_format ='yxyx',
            boxes2_src_format ='yxyx',
            mask_val = 0.):

        self.mode = mode
        self.is_aligned = is_aligned
        self.boxes1_src_format = boxes1_src_format
        self.boxes2_src_format = boxes2_src_format
        'optional'
        self.use_masking = use_masking
        self.mask_val = mask_val
        'basic settings'
        self.eps = keras.backend.epsilon()
        self.ciou_factor = 4/math.pi**2


        if self.boxes1_src_format not in self.valid_boxes_src_format:
            raise ValueError(f'boxes1_src_format : {self.boxes1_src_format} is invalid format'
                            f'please check support list { self.valid_boxes_src_format}  @{self.__class__.__name__}'
            )
        
        if self.boxes1_src_format !='yxyx':
            self.boxes1_format_converter = BBoxesFormatTransform(
                                            convert_type =boxes1_src_format+'2yxyx'  
            )
        if self.boxes2_src_format not in self.valid_boxes_src_format:
            raise ValueError(f'boxes1_src_format : {self.boxes2_src_format} is invalid format'
                            f'please check support list { self.valid_boxes_src_format}  @{self.__class__.__name__}'
            )    
    
        if self.boxes2_src_format !='yxyx':
            self.boxes2_format_converter = BBoxesFormatTransform(
                                        convert_type =boxes2_src_format+'2yxyx'
            )


    def masked_iou(self, boxes1, boxes2, iou):
        if not self.is_aligned :
            boxes1 = boxes1[...,None,:]
            boxes2 = boxes2[...,None,:,:]
        boxes1_mask = tf.math.less_equal(tf.math.reduce_max(boxes1,axis=-1),0.) #
        boxes2_mask = tf.math.less_equal(tf.math.reduce_max(boxes2,axis=-1),0.)
        background_mask = tf.math.logical_or(boxes1_mask, boxes2_mask)
        masked_iou = tf.where(background_mask, self.mask_val, iou)
        return masked_iou

    def giou(self,iou,union_area,enclose_hw):

        enclose_area = tf.maximum(tf.math.reduce_prod(enclose_hw, axis=-1), self.eps )
        giou = iou - (enclose_area-union_area)/enclose_area
        return giou

    def siou(self,iou, ctr_distance, enclose_hw, hw_boxes1, hw_boxes2):
        siou_theta =  4.0
        sigma_cw = ctr_distance[...,1] + self.eps 
        sigma_ch = ctr_distance[...,0] + self.eps 
        ctr_distance_squared =tf.reduce_sum(tf.math.square(ctr_distance), axis=-1)
        sigma = tf.pow(ctr_distance_squared, 0.5)
        sin_alpha = tf.abs(sigma_ch) / sigma
        sin_beta = tf.abs(sigma_cw) / sigma
        sin_alpha = tf.where(tf.less_equal(sin_alpha, tf.math.sin(math.pi/4)),sin_alpha,sin_beta)
        angle_cost = tf.math.cos( tf.math.asin(sin_alpha)*2 -math.pi/2)
        rho_x = (sigma_cw / enclose_hw[1])**2  # ρ_x
        rho_y = (sigma_ch / enclose_hw[0])**2  # ρ_y
        gamma = 2 - angle_cost  # γ
        distance_cost = (1 - tf.math.exp(-1 * gamma * rho_x)) +(1 - tf.math.exp(-1 * gamma * rho_y))
        # hw_boxes1 = boxes1[...,2:]-boxes1[...,:2] #(b,m,1,2) or (b,m,2)
        # hw_boxes2 = boxes2[...,2:]-boxes2[...,:2] #(b,1,n,2) or (b,n,2)

        omiga_w = tf.abs(hw_boxes1[1] - hw_boxes2[1]) / tf.math.maximum(hw_boxes1[1], hw_boxes2[1])  # ω_w
        omiga_h = tf.abs(hw_boxes1[0] - hw_boxes2[0]) / tf.math.maximum(hw_boxes1[0], hw_boxes2[0])  # ω_h
        shape_cost = tf.pow(1 - tf.math.exp(-1 * omiga_w), siou_theta)
        shape_cost += tf.pow(1 - tf.math.exp(-1 * omiga_h), siou_theta)
        siou = iou - ((distance_cost + shape_cost) * 0.5)
        return siou

    def ciou(self,iou,ctr_distance,enclose_hw,hw_boxes1,hw_boxes2):
        #factor=4/math.pi**2
        # hw_boxes1 = boxes1[...,2:]-boxes1[...,:2] #(b,m,1,2) or (b,m,2)
        # hw_boxes2 = boxes2[...,2:]-boxes2[...,:2] #(b,1,n,2) or (b,n,2)
        arctan_wh_boxes1 = tf.math.atan2(hw_boxes1[...,1], tf.maximum(hw_boxes1[...,0], self.eps ))  #(b,m,1) or (b,m)
        arctan_wh_boxes2 = tf.math.atan2(hw_boxes2[...,1], tf.maximum(hw_boxes2[...,0], self.eps ))  #(b,1) or (b,n)
        v = self.ciou_factor*tf.pow( arctan_wh_boxes1 - arctan_wh_boxes2, 2) #(b,1,n,2) or (b,n,2)
        alpha = v/(v-iou+(1+self.eps ))
        ''
        ctr_distance_squared = ctr_distance_squared =tf.reduce_sum(tf.math.square(ctr_distance), axis=-1)
        enclose_diagonal_squared = tf.reduce_sum(tf.math.square(enclose_hw),axis=-1)+self.eps 

        ciou = iou - (ctr_distance_squared/enclose_diagonal_squared+ v*alpha)
        return ciou

    def apply(self,boxes1, boxes2):

        intersect_area = compute_intersection(boxes1, boxes2, self.is_aligned) #(b,m,n) or (b,m)
        boxes1_area = compute_area(boxes1)   #(b,m)
        boxes2_area = compute_area(boxes2)   #(b,n) or (b,m)

        'base ious'
        union_area = boxes1_area+boxes2_area if self.is_aligned else boxes1_area[..., None]+boxes2_area[..., None, :]
        union_area = tf.maximum(union_area-intersect_area,self.eps ) #(b,m,n) or (b,m)
        iou = intersect_area/union_area #(b,m,n) or (b,m)
        if self.mode=='iou':
            return iou

        'ciou'
        if not self.is_aligned :
            boxes1 = boxes1[...,None,:]
            boxes2 = boxes2[...,None,:,:]

        yx_min = tf.minimum(boxes1[...,:2], boxes2[...,:2])  #maximum((b,m,2),(b,m,2)) => (b,m,2)
        yx_max = tf.maximum(boxes1[...,2:], boxes2[...,2:])  #maximum((b,m,2),(b,m,2)) => (b,m,2)
        enclose_hw = tf.maximum(yx_max-yx_min, 0.)#(b,m,n,2)or (b,m,2)


        if self.mode=='giou':
            return self.giou(iou,union_area,enclose_hw)

        #enclose_diagonal_squared = tf.reduce_sum(tf.math.square(enclose_hw),axis=-1)+self.eps  #(b,m,n,2)or(b,m,2)
        ctr_distance =  (boxes1[...,2:]+boxes1[...,:2])/2-(boxes2[...,2:]+boxes2[...,:2])/2 #(b,m,n,2)or(b,m,2)
        #ctr_distance_squared =tf.reduce_sum(tf.math.square(ctr_distance), axis=-1)
        hw_boxes1 = boxes1[...,2:]-boxes1[...,:2] #(b,m,1,2) or (b,m,2)
        hw_boxes2 = boxes2[...,2:]-boxes2[...,:2] #(b,1,n,2) or (b,n,2)

        if self.mode=='siou':
            'need ctr_distance and enclose_hw'
            return self.siou(iou, ctr_distance, enclose_hw, hw_boxes1, hw_boxes2)

        if self.mode=='ciou':
            'need ctr_distance and enclose_hw'
            return self.ciou(iou,ctr_distance,enclose_hw,hw_boxes1,hw_boxes2)

        raise ValueError("invalid iou mode, it must be 'iou','ciou','giou' or 'siou' "
                        f"but got {self.mode} @{self.__class__.__name__}")

    def __call__(self, boxes1, boxes2):
        """ 
        boxes1 (tensor) : [b,m,4] , [m,2]
        boxes2 (tensor) : [b,n,4] , [n,2]
        """
        
        if not isinstance(boxes1, tf.Tensor) or not isinstance(boxes2, tf.Tensor):
            raise TypeError(
                f"input data (boxes1, boxes2) type must be Tensor, "
                f"but got < boxes1 : {type(boxes1)}  and  boxes1 : {type(boxes1)} >"
                f"@{self.__class__.__name__} "
            )
        'prepare data by converting bboxes to yxyx format'
        if hasattr(self, 'boxes1_format_converter'):
            boxes1 = self.boxes1_format_converter(boxes1)
        if hasattr(self, 'boxes2_format_converter'):
            boxes2 = self.boxes2_format_converter(boxes2)

        iou = self.apply(boxes1, boxes2)
        return self.masked_iou(boxes1,boxes2,iou)if self.use_masking else iou
    

