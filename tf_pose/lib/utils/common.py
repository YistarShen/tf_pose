from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import tensorflow as tf
import matplotlib.pyplot as plt
import os, cv2
import numpy as np

############################################################################
#
# 
############################################################################
# def is_path_avaiable(path, RuntimeError_Flag=True):
#     if not os.path.exists(path):
#         if RuntimeError_Flag :
#             raise RuntimeError("Warrning !!!! no such dir : \n{}\n".format(path)) 
#         else:
#             return print("Warrning !!!! no such dir : \n{}\n".format(path))
#     else:
#         return print("successfully connect floder  : \n{}\n".format(path))

def is_path_available(path, RuntimeError_Flag=True):
    if not os.path.exists(path):
        if RuntimeError_Flag :
            raise RuntimeError(
                "Warrning !!!! no such dir : \n{}\n".format(path)
            ) 
        else:
            print("Warrning !!!! no such dir : \n{}\n".format(path))
            return False
    else:
        print("successfully connect floder  : \n{}\n".format(path)) 
        return True   
    
############################################################################
#
# 
############################################################################
def convert_images_into_np_array(img_jpg) ->List[np.ndarray]:
    images_list = []
    for img_name in img_jpg :
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image).reshape( ( image.shape[0], image.shape[1], 3)).astype(np.uint8)
        images_list.append(image.copy())   
        print(f"{img_name} : {image.shape}")
        del image
    return images_list
    
############################################################################
#
# 
############################################################################
'videos property setting'
def get_video_property(video_path, cv2_rot_k=0):
    cap = cv2.VideoCapture(video_path)
    cam_res_w  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    cam_res_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    fps = cap.get(cv2.CAP_PROP_FPS)
    while(cap.isOpened()):
        was_read, src_frame = cap.read()
        src_frame = cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB)
        assert src_frame.ndim == 3, "each frame  must be ndim =3!!!!!"
        if cv2_rot_k != 0 :
            src_frame = tf.image.rot90(src_frame, k = cv2_rot_k)  #(1920,1080,3)
        #src_frame = cv2.rotate(src_frame, cv2.ROTATE_90_CLOCKWISE)
        cam_res_w  =  src_frame.shape[1] # float `width`
        cam_res_h = src_frame.shape[0]  # float `height`
        plt.figure(figsize=(10,10))
        plt.title(f'Image \n ( fps : {int(fps)},  Res_hw : {cam_res_h}x{cam_res_w} )')
        plt.imshow(src_frame)
        plt.show()
        break
    return cam_res_w, cam_res_h, fps, cv2_rot_k

############################################################################
#
# 
############################################################################
def merge_dicts(parent_cfg, cfg):
    assert isinstance(parent_cfg, dict) and isinstance(cfg, dict), \
    f"parent_cfg and cfg must both are dict, but got {type(parent_cfg)} and {type(cfg)} "
    updated_cfg = {**parent_cfg, **cfg }
    return updated_cfg

############################################################################
#
# 
############################################################################
def tf_img_norm_transform(img, inv=False):
    img_mean = tf.constant([0.485, 0.456, 0.406],dtype=tf.float32)
    img_std = tf.constant([0.229, 0.224, 0.225],dtype=tf.float32)
    if (inv==False):
        img = img / 255.0
        img = (img - img_mean)/img_std
    else:
        img =  img*img_std + img_mean
    return img
