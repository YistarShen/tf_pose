from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from tensorflow import Tensor
import tensorflow as tf
import numpy as np
import sys, os, cv2
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------
#
#
#--------------------------------------------------------------------------------------------
def mvlaDet_ImageDemoV1(upload : List,  
                        mvlaDet_Infer : object):
    
    def convert_images_into_np_array(img_jpg):
        images_list = []
        for img_name in img_jpg :
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image).reshape( ( image.shape[0], image.shape[1], 3)).astype(np.uint8)
            images_list.append(image)   
            print(image.shape)
            del image
        return images_list 
       
    ' Verify inference APIs'
    assert isinstance(mvlaDet_Infer, Callable),'mvlaDet_Infer is not Callable '
    
    'convert image to np array for image plotting by cv2 lib'
    image_np_list = convert_images_into_np_array(upload)
    for src_image_np in image_np_list :
        'Inference of mvlaDet --------------------------------START'
        '''
        #bbox_yx_SrcImg, decoded_detections = mvlaDet_Infer(src_image_np)
        bbox_yx_SrcImg = mvlaDet_Infer(src_image_np)
        '''
        dict_out = mvlaDet_Infer(src_image_np)
        
        
        'Inference of mvlaDet -------------------------------- END'

        bbox_yx_SrcImg = dict_out['bbox_yx']


        'show img with predicted bbox'
        top, left, bottom, right=  bbox_yx_SrcImg[0,...].numpy()
        cv2.rectangle(src_image_np, (int(left), int(top)), (int(right), int(bottom)), 
                      color=(255,0,0), thickness=10)
    
        plt.figure(figsize=(10,10)) 
        plt.title(f'image with personal bbox @ src_img frame')
        plt.imshow(src_image_np)
        plt.show()


#--------------------------------------------------------------------------------------------
#
#
#--------------------------------------------------------------------------------------------
def draw_bboxes_with_tf_nmsed_decode(batch_imgs : tf.Tensor,
                                    tf_nmsed_decode : object) :
    
    r"""draw_bboxes_with_tf_nmsed_decode
    inputs:
        imgs = 
        tf_nmsed_decode = 

    """
    batch_size = tf.shape(batch_imgs)[0]


    # images_shape = tf.cast(tf.shape(batch_imgs), dtype=tf.float32)
    # images_shape = images_shape[1:3]
  
    batch_bboxes = tf_nmsed_decode.nmsed_boxes
    batch_scores = tf_nmsed_decode.nmsed_scores
    batch_cls = tf_nmsed_decode.nmsed_classes
    batch_num_detections = tf_nmsed_decode.valid_detections

    # colors = tf.constant([[255, 0, 0]], dtype=tf.float32)
    # imgs_with_bb = tf.image.draw_bounding_boxes(batch_imgs, batch_bboxes, colors)

    plt.figure(figsize=(40,40))
    for idx in range(batch_size):
        if idx == 16:
            break

        image = tf.cast(batch_imgs[idx],dtype=tf.int32).numpy()
        bboxes = batch_bboxes[idx]
        num_detections = batch_num_detections[idx]
        labels = batch_cls[idx]
        plt.subplot(4,4,idx+1)
        plt.imshow(image)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        for i in range(num_detections):
            y1, x1, y2, x2 = tf.split(bboxes[i, :], 4)
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            patch = plt.Rectangle(
                [x1, y1], w, h, fill=False, edgecolor=[1, 0, 0], linewidth=1.5
            )
            ax.add_patch(patch)
            label_text = '{}'.format(int(labels[i]))
            color=[0, 0, 1]
            plt.text(x1, 
                     y1, 
                     label_text, 
                     fontsize=15, 
                     bbox={"facecolor": color, "alpha": 0.4}
            )
        plt.title("detModel pred", fontsize = 20)
    plt.show()

