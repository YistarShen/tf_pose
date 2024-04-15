import tensorflow as tf

#----------------------------------------------------------------
# 
#-----------------------------------------------------------------
def tf_clean_kps_outside_bbox(
        src_kps, bbox_xywh, sigma=0.
):
 
    compute_dtype = src_kps.dtype
    eff_margin = tf.cast( sigma*3., dtype=compute_dtype)
    # cond_max = tf.math.less_equal(
    #     src_kps[...,:2], bbox_cxcywh[0:2]+bbox_cxcywh[2:]/2+eff_margin
    # )   #(17,2)
    # cond_min = tf.math.greater_equal( 
    #     src_kps[...,:2], bbox_cxcywh[0:2]-bbox_cxcywh[2:]/2-eff_margin
    # ) #(17,2)

    cond_max = tf.math.less_equal(
        src_kps[...,:2], 
        bbox_xywh[...,None,:2]+bbox_xywh[...,None,2:] +eff_margin
    )   #(b,17,2)
    cond_min = tf.math.greater_equal( 
        src_kps[...,:2], 
        bbox_xywh[...,None,:2]- eff_margin
    ) #(b,17,2)
    cond = tf.concat(
        [cond_max,cond_min], axis=-1
    ) #(b,17,4)
    cond = tf.math.reduce_all(
        cond, axis=1
    )  #(b,17,)
    cal_src_kps = tf.where(
        tf.expand_dims(cond, axis=-1), 
        src_kps, 
        tf.cast(0., dtype=compute_dtype)
    )  #(b,17,3)
    return cal_src_kps

#----------------------------------------------------------------
# 
#-----------------------------------------------------------------
def tf_kps_clip_by_img(kps, img_shape_yx):

    """
    kps : (17,3)  => (17, [x,y,vis])
    img_shape_xy : (w,h)
     """
    cond_max = tf.math.less_equal( 
        kps[...,:2], [img_shape_yx[1], img_shape_yx[0]]
    )
    cond_min = tf.math.greater_equal( 
        kps[...,:2], [0., 0.]
    )
    vis = tf.expand_dims(
        tf.greater(kps[...,2], 0),axis=-1
    )
    cond = tf.concat(
        [cond_max, cond_min, vis], axis=-1
    )  #(17,5)
    cond = tf.math.reduce_all(
        cond, axis=-1, keepdims=True
    ) #(17,1)
    cal_kps = tf.where(
        cond, kps[...,:2], tf.cast(0., dtype=kps.dtype)
    )        #(17,2)
    cal_vis = tf.cast(
        cond, dtype=cal_kps.dtype
    )        #(17,1)
    cal_kps = tf.concat(
        [cal_kps, cal_vis],axis=-1
    )
    return cal_kps