

1. 

   
    'image': TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name=None), 
    'bbox': TensorSpec(shape=(4,), dtype=tf.float32, name=None), 
    'kps': TensorSpec(shape=(17, 3), dtype=tf.float32, name=None), 
    'image_size': TensorSpec(shape=(2,), dtype=tf.int32, name=None), 
    'bbox_format': TensorSpec(shape=(), dtype=tf.string, name=None), 
    'meta_info': {'src_image': TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name=None), 
                  'src_height': TensorSpec(shape=(), dtype=tf.int64, name=None), 
                  'src_width': TensorSpec(shape=(), dtype=tf.int64, name=None), 
                  'src_keypoints': TensorSpec(shape=(None,), dtype=tf.float32, name=None), 
                  'src_bbox': TensorSpec(shape=(4,), dtype=tf.float32, name=None), 
                  'src_num_keypoints': TensorSpec(shape=(), dtype=tf.int64, name=None), 
                  'area': TensorSpec(shape=(), dtype=tf.float32, name=None), 
                  'category_id': TensorSpec(shape=(), dtype=tf.int64, name=None), 
                  'id': TensorSpec(shape=(), dtype=tf.int64, name=None), 
                  'image_id': TensorSpec(shape=(), dtype=tf.int64, name=None), 
                  'iscrowd': TensorSpec(shape=(), dtype=tf.int64, name=None)
    }, 
    'transform2src': {'scale_xy': TensorSpec(shape=(2,), dtype=tf.float32, name=None), 
                  'pad_offset_xy': TensorSpec(shape=(2,), dtype=tf.float32, name=None), 
                  'bbox_lt_xy': TensorSpec(shape=(2,), dtype=tf.float32, name=None)
    }