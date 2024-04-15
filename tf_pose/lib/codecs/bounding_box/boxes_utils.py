import tensorflow as tf


#-----------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------  
def boxes_format_convert(
                boxes : tf.Tensor, 
                box_src_format : str = "xywh", 
                box_target_format : str = "xyxy"):

    if box_src_format == box_target_format :
        return boxes
    
    convert_type = box_src_format + "2" + box_target_format 
    if  (convert_type=="cxcywh2yxyx"):
        boxes = tf.concat(
            [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0], axis=-1
        ) #xy_min,xy_max
        boxes = tf.stack(
            [boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]],  axis=-1
        ) # yx_min, yx_max
    if  (convert_type=="cxcywh2xyxy"):       
        boxes = tf.concat(
            [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0], axis=-1
        ) #xy_min,xy_max
    elif  (convert_type=="cxcywh2xywh"):
        boxes = tf.concat(
            [ (boxes[..., :2] - boxes[..., 2:]/2.0), boxes[..., 2:] ], axis=-1
        ) #cxcy,wh  
    elif (convert_type=="yxyx2cxcywh"):
        boxes = tf.concat(
            [ (boxes[..., :2] + boxes[..., 2:])/2.0, (boxes[..., 2:] - boxes[..., :2]) ], axis=-1
        ) #cycx,hw
        boxes = tf.stack(
            [boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1
        ) # cxcy,wh
    elif (convert_type=="yxyx2xyxy"):
        boxes = tf.stack(
            [boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1
        ) # xyxy

    elif (convert_type=="yxyx2xywh"):
        boxes = tf.stack(
            [boxes[..., 1], boxes[..., 0], boxes[..., 3]-boxes[..., 1], boxes[..., 2]-boxes[..., 0]], axis=-1
        ) # xyxy
    elif (convert_type=="xyxy2xywh"):
        boxes = tf.concat(
            [boxes[..., :2] , boxes[..., 2:] - boxes[..., :2]],axis=-1
        ) # xy_min,wh
    elif (convert_type=="xyxy2yxyx"): 
        boxes = tf.stack(
            [boxes[..., 1] , boxes[..., 0], boxes[..., 3] , boxes[..., 2]],axis=-1
        ) # xy_min,wh
    elif (convert_type=="xyxy2cxcywh"): 
        boxes = tf.concat(
            [ (boxes[..., :2] + boxes[..., 2:])/2.0, (boxes[..., 2:] - boxes[..., :2]) ], axis=-1
        ) #cycx,hw     
    elif (convert_type=="xywh2xyxy"):
        boxes = tf.concat(
            [boxes[..., :2], boxes[..., :2] + boxes[..., 2:]],axis=-1
        ) #xy_min,xy_max
    elif (convert_type=="xywh2yxyx"):
        boxes = tf.concat(
            [boxes[..., :2], boxes[..., :2] + boxes[..., 2:]],axis=-1
        ) #xy_min,xy_max 
        boxes = tf.stack(
            [boxes[..., 1] , boxes[..., 0], boxes[..., 3] , boxes[..., 2]],axis=-1
        ) # xy_min,wh    
    elif  (convert_type=="xywh2cxcywh"):
        boxes = tf.concat(
            [ (boxes[..., :2] + boxes[..., 2:]/2.0), boxes[..., 2:] ], axis=-1
        ) #cxcy,wh         
    else:
        r'''
        support_convert_type = ["cxcywh2yxyx","cxcywh2xyxy","cxcywh2xywh",
                                "yxyx2cxcywh","yxyx2xyxy","yxyx2xywh",
                                "xyxy2xywh","xyxy2yxyx","xyxy2cxcywh",
                                "xywh2xyxy","xywh2yxyx","xywh2cxcywh"]
        '''
        tf.debugging.Assert(
            False, [f"unknown conveter TYPE : {convert_type}"]
        )

    return boxes


#----------------------------------------------------------------
# 
#-----------------------------------------------------------------
def fix_bboxes_aspect_ratio(
        bboxes_xywh : tf.Tensor, aspect_ratio_xy : float
):  
    """ 
    aspect_ratio_xy = image_size_xy[0]/image_size_xy[1]
    """

    bboxes_cxcywh = boxes_format_convert(bboxes_xywh, 'xywh', 'cxcywh') #(b,4) or (b,None,4)
    aspect_ratio_xy = tf.cast(aspect_ratio_xy, dtype=bboxes_xywh.dtype)
    
    W = bboxes_cxcywh[...,2]  #(b,) or (b,None)
    H = bboxes_cxcywh[...,3]
    cond = tf.greater(
         W/H, aspect_ratio_xy
    ) #(b,) or (b,None)
    bbox_scale_wh = tf.where(
        cond,[ W, W/aspect_ratio_xy], [H*aspect_ratio_xy,H]
    )#(b,2) or (b,None)

    bbox_scale_wh = tf.where(
        cond[...,None],
        tf.stack([W, W/aspect_ratio_xy], axis=-1),
        tf.stack([H*aspect_ratio_xy, H], axis=-1)
    )#(b,2) or (b,None,2)

    
    bbox_xywh = tf.concat(
            [bboxes_cxcywh[...,:2]- bbox_scale_wh/2., bbox_scale_wh], axis=-1
    ) #(b,4) or (b,None,4)
    return bbox_xywh 


#-----------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------  
def dist2bbox(distance_xyxy : tf.Tensor, 
              anchor_points  : tf.Tensor
    ):
    r"""Decodes distance to anchor'center into xyxy boxes.

    Args:
        anchor_points:
        distance_xyxy:
    Returns:
        bbox_xyxy

    dist2bbox https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/object_detection/yolo_v8/yolo_v8_detector.py
    """
    dist2lt, dist2rb = tf.split(distance_xyxy, 2, axis=-1) #(b,8400,4)=>(b,8400,2)&(b,8400,2)
    anchor_points = tf.cast(anchor_points, dtype=distance_xyxy.dtype)
    left_top_xy = anchor_points - dist2lt  #(b,8400,2)
    right_bottom_xy = anchor_points + dist2rb #(b,8400,2)
    return tf.concat([left_top_xy, right_bottom_xy], axis=-1)  # xyxy bbox

#-----------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------  
def bbox2dist(bboxes_xyxy : tf.Tensor, 
              anchor_points  : tf.Tensor, 
              reg_max : int=16
    ):
    r"""Decodes distance to anchor'center into xyxy boxes.

    Args:
        anchor_points:
        distance_xyxy:
    Returns:
        bbox_xyxy
    """
   
    lt, rb = tf.split(bboxes_xyxy, 2, axis=-1) #(b,8400,4)=>(b,8400,2)&(b,8400,2)
    anchor_points = tf.cast(anchor_points, dtype=bboxes_xyxy.dtype )
    dist2lt = anchor_points - lt
    dist2rb = anchor_points + rb  #(b,8400,2)
    dist_xyxy = tf.concat([dist2lt, dist2rb], axis=-1)  # xyxy bbox
    return tf.clip_by_value(
            dist_xyxy, 0.,  tf.cast(reg_max-1, dtype=dist_xyxy.dtype)-0.01
        )

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def create_bounding_box_dataset(
        image_size = 640,
        num_class= 5,
        label_start_idx = 1, 
        dtype=tf.float32):
    
    scale = tf.cast(image_size, dtype=dtype)
    cls_1  =  tf.constant( [1,2,3],dtype=tf.int32)
    cls_2  = tf.constant( [2,num_class,0],dtype=tf.int32)
    batch_label = tf.stack([cls_1,cls_2],axis=0)
    boxes_1 = tf.constant(
        [
            [0.1, 0.1, 0.5, 0.4],
            [0.67, 0.75, 0.23, 0.23],
            [0.25, 0.25, 0.6, 0.35],
        ],
        dtype=dtype,
    )
    boxes_2 = tf.constant(
       [
            [0.5, 0.4, 0.3, 0.25],
            [0.25, 0.25, 0.23, 0.23],
            [0., 0., 0., 0.]
       ],
        dtype=dtype,
    )
    batch_boxes = tf.stack([boxes_1,boxes_2], axis=0)*scale

    data = dict()
    data['bbox'] = batch_boxes
    data['labels'] = batch_label
    return data 

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def create_dummy_preds(
        image_size = 640,
        num_class= 5,
        dtype = tf.dtypes.float32):
    
    feats_size = [image_size//8, image_size//16, image_size//32]
    feat_80x80 = tf.random.uniform(
        shape=(2,feats_size[0]**2, 4),
        minval=0,
        maxval=feats_size[0]//2, dtype=dtype,
    )
    feat_40x40 = tf.random.uniform(
        shape=(2,feats_size[1]**2,4),
        minval=0,
        maxval=feats_size[1]//2, dtype=dtype,
    )
    feat_20x20 = tf.random.uniform(
        shape=(2,feats_size[2]**2,4),
        minval=0,
        maxval=feats_size[2]//2, dtype=dtype,
    )

    pred_bbox = tf.concat([feat_80x80,feat_40x40,feat_20x20],axis=1)

    pred_class = tf.random.uniform(
        shape=(2,8400,num_class),
        minval=0.,
        maxval=1., dtype=dtype,
    )
    return pred_bbox,  pred_class


