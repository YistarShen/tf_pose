import tensorflow as tf
#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def get_anchor_points(
        image_shape,
        strides=[8, 16, 32],
        dtype = tf.float32,
    ):
    """ 
    Args:
        image_shape: tuple or list of two integers representing the height and
            width of input images, respectively.
        strides: tuple of list of integers, the size of the strides across the
            image size that should be used to create anchors.

    Returns:
        A tuple of anchor centerpoints and anchor strides. Multiplying the
        two together will yield the centerpoints in absolute x,y format.
    """
    all_anchors = []
    all_strides = []
    for stride in strides:
        grid_coords_y = tf.cast(tf.range(0, image_shape[0],stride)+stride//2, dtype=dtype) #(grid_h,)
        grid_coords_x = tf.cast(tf.range(0, image_shape[1],stride)+stride//2, dtype=dtype)
        grid_x, grid_y = tf.meshgrid(grid_coords_x, grid_coords_y)  # (grid_h,grid_w) and (grid_h,grid_w)
        anchor_ctr_xy = tf.stack([grid_x, grid_y], axis=-1)      # (grid_h,grid_w,2)
        anchor_ctr_xy = tf.reshape(anchor_ctr_xy,[-1,2])
        all_anchors.append(anchor_ctr_xy)    # [(grid_h*grid_w,2),...]
        all_strides.append(tf.tile(tf.cast([stride], dtype=dtype),[anchor_ctr_xy.shape[0]]))

    all_strides = tf.concat(all_strides, axis=0)    #(8400,)
    all_strides = tf.expand_dims(all_strides, axis=-1) #(8400,)=>(8400,1)
    all_anchors = tf.concat(all_anchors, axis=0)    #(8400,2)
    all_anchors = all_anchors / all_strides      #(8400,2)/(8400,1) =>(8400,2)
    return all_anchors, all_strides

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def is_anchor_center_within_box(
        anchor_points_xy,
        gt_bboxes_xyxy,
    ):
    """
        anchor_points_xy : (8400,2) at image_frame
        gt_bboxes_xyxy : (b,num_gt_bbox,4) with xyxy format at image_frame
        return : (b,num_gt_bbox,8400)
    """

    lt_cond = tf.less(
        gt_bboxes_xyxy[...,None,:2], anchor_points_xy
    )
    rb_cond = tf.greater(
        gt_bboxes_xyxy[...,None,2:], anchor_points_xy
    )
    mask_in_gt_boxes = tf.math.reduce_all(
        tf.concat([lt_cond,rb_cond],axis=-1),axis=-1
    )
    return tf.cast(mask_in_gt_boxes, dtype=gt_bboxes_xyxy.dtype)