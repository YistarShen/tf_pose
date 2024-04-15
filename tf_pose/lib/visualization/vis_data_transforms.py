from typing import Dict, List, Optional, Tuple, Union, Any, Sequence, Callable
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
import matplotlib.pyplot as plt
from typing import Dict, List, Union,Tuple
from lib.datasets.transforms.utils import PackInputTensorTypeSpec
from .base_vis import BasePoseVisFun
#---------------------------------------------------------------------------
#
#---------------------------------------------------------------------------

class Vis_SampleTransform(BasePoseVisFun):
    VERSION = '1.0.0'

    R"""  Vis_SampleTransform
    
    i.e. :
        batched_samples_list = [sample for sample in batch_dataset.take(2)]
        vis_fn = Vis_SampleTransform(
            figsize = (16, 8),
            sel_batch_ids  = [0]
        )
        vis_fn(
            batched_samples_list = batched_samples_list
        )

    """
    def __init__(
            self,
            plot_transformed_bbox : bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.plot_transformed_bbox = plot_transformed_bbox

    def parse_dataset(
        self, 
        batched_samples_list : List[dict], 
    ) :

        batched_src_samples_list = [samples.pop('meta_info', None) for samples in batched_samples_list]
        data_dict = dict(
            img = self.extract_key_samples(batched_samples_list, 'image'),
            kpts = self.extract_key_samples(batched_samples_list, 'kps'),
            bboxes = self.extract_key_samples(batched_samples_list, 'bbox'),
            labels = self.extract_key_samples(batched_samples_list, 'labels'),
            src_img = self.extract_key_samples(batched_src_samples_list, 'src_image'),
            src_bboxes = self.extract_key_samples(batched_src_samples_list, 'src_bbox'),
            src_labels = self.extract_key_samples(batched_src_samples_list, 'src_labels'),
            src_img_id = self.extract_key_samples(batched_src_samples_list, 'image_id'),
            src_ids = self.extract_key_samples(batched_src_samples_list, 'id'),
            src_kpts = self.extract_key_samples(batched_src_samples_list, 'src_keypoints'),
        )
        del batched_samples_list
        return data_dict
    

    def call(
        self, 
        img : Tensor,
        kpts : Optional[Tensor] = None,
        bboxes : Optional[Tensor] = None,
        labels : Optional[Tensor] = None,
        src_img : Optional[Tensor] = None,
        src_kpts : Optional[Tensor] = None,
        src_bboxes : Optional[Tensor] = None,
        src_labels : Optional[Tensor] = None,
        src_img_id : Optional[Tensor] = None,
        src_ids : Optional[Tensor] = None,
        **kwargs
    )->None:
        
        r"""

        """
        'input data for plotting'
        np_resized_img = img.numpy() # to narray for plotting
        np_src_img = None if src_img is None else src_img.numpy()   # to narray for plotting
        image_id = '' if src_img_id is None else src_img_id 
        instance_ids = src_ids  if (isinstance(src_ids,tf.Tensor) and src_ids.shape.rank==0) else '' #only show instance_ids in single pose 

        'setup plot figure'
        sub_plots = 2 if np_src_img is not None else 1
        plt.figure(figsize=self.figsize)

        self.base_plot(
            title = f"transform_image",
            image = np_resized_img,
            kpts = kpts,
            labels = labels,
            bboxes = bboxes if self.plot_transformed_bbox else None,
            subplot_spec = (1,sub_plots,1)
        )
        if np_src_img is not None :
            text =  f"\n ( image_id : {image_id }  instance_id : {instance_ids} )"
            self.base_plot(
                title = f"src_image" + text,
                image = np_src_img,
                kpts = src_kpts,
                bboxes = src_bboxes,
                labels = src_labels,
                subplot_spec = (1,sub_plots,2)
            )


# #---------------------------------------------------------------------------
# #
# #---------------------------------------------------------------------------
# def Vis_PoseSampleTransform(
#     batched_samples: Union[Dict, List],    
#     batch_ids :  Union[List[int], int] = [0],
#     plot_transformed_bbox : bool = True,
#     figsize : Tuple[int] = (20,10)
# ):
#     if not isinstance(batched_samples, (dict, list)):
#         raise TypeError(
#             "input samples must be dict or List"
#         )
#     if isinstance(batched_samples, dict):
#        batched_samples = [batched_samples]
#     if isinstance(batch_ids, int):
#        batch_ids = [batch_ids]

#     def plot(
#             title, image, bboxes,  labels=None, kpts=None
#     ):  
#         plt.title(
#             title, fontsize = 12
#         )
#         plt.imshow(image)
#         ith_obj = -1 
#         for i, bbox in enumerate(bboxes):   
#             label = labels[i] if labels is not None else 1
#             if label==1: 
#                 color = np.random.uniform(
#                     low=0., high=1., size=3
#                 )
#                 ith_obj +=1

#             ax = plt.gca() 
#             x1, y1, w, h = bbox
#             patch = plt.Rectangle(
#                 [x1, y1], w, h, 
#                 fill=False, 
#                 edgecolor=color, 
#                 linewidth=2
#             )
#             ax.add_patch(patch)
#             text = "{}".format(label)
#             ax.text(
#                 x1, y1, text,
#                 bbox={"facecolor": color, "alpha": 0.8},
#                 clip_box=ax.clipbox,
#                 clip_on=True,
#             )
#             if kpts is None :
#                 continue
#             kpt_ith = kpts[ith_obj]
#             for j in range(0,kpt_ith.shape[0]):
#                 kps_x = int((kpt_ith[j,0]))
#                 kps_y = int((kpt_ith[j,1]))
#                 plt.scatter(kps_x,kps_y, color=color)




#     if batched_samples[0]['image'].shape.rank==3:
#         raise ValueError(
#             f"Only support batched sample "
#         ) 
#     if batched_samples[0]['bbox'].shape.rank==2:
#         is_single_pose = True 
#     elif batched_samples[0]['bbox'].shape.rank==3:
#         is_single_pose = False 
#     else:
#         raise ValueError(
#             f"features['bbox'].shape must be (b,n,4) or (b,4), "
#             f"but got {features['bbox'].shape}"
#         ) 
    
#     PackInputTensorTypeSpec(
#         batched_samples[0],{}, show_log=True
#     )
#     print("\n\n\n")
#     for features in batched_samples :
#         for batch_id in batch_ids :
#             image = features["image"][batch_id].numpy()
#             bboxes = features['bbox'][batch_id]

#             if features.get('labels',None) is not None :                         
#                 labels = features['labels'][batch_id]

#             if features.get('kps',None) is not None :
#                 kpts = features['kps'][batch_id]  

#             if  is_single_pose:  
#                 bboxes = tf.expand_dims(bboxes, axis=0) 
#                 if 'kpts' in locals() : 
#                     kpts = tf.expand_dims(kpts, axis=0) 

#             if  features.get('meta_info',None) is not None :
#                 meta_info = features['meta_info']
#                 image_id = meta_info['image_id'][batch_id]
#                 instance_id = meta_info['id'][batch_id]
#                 src_image = meta_info['src_image'][batch_id].numpy()
#                 src_bbox = tf.reshape(
#                     meta_info['src_bbox'][batch_id],[-1,4]
#                 )
#                 if 'labels' in locals() : 
#                     src_labels = meta_info['src_labels'][batch_id] 
#                 if 'kpts' in locals() : 
#                     src_kps = tf.reshape(
#                         meta_info['src_keypoints'][batch_id],[-1,*(kpts.shape[1:])]
#                     )
            
#             sub_plots = 2 if 'meta_info' in locals() else 1
#             plt.figure(figsize=figsize)
#             plt.subplot(1,sub_plots, 1)

#             text  =  f" in {'single_pose' if  is_single_pose else 'multi_poses'} DS"
#             if 'meta_info' in locals() :
#                 #text +=  f"---( {'instance_id : ' if is_single_pose else 'image_id : ' }"
#                 text +=  f"\n ( image_id : {image_id }  "
#                 text +=  f"instance_id : {instance_id  if is_single_pose else '' } )"
                
#             plot(
#                 "transform_image" + text,
#                 image,
#                 bboxes if plot_transformed_bbox else tf.zeros_like(bboxes), 
#                 labels if 'labels' in locals() else None, 
#                 kpts if 'kpts' in locals() else None,
#             )

#             if 'meta_info' not in locals() :   
#                 return
#             plt.subplot(1,sub_plots, 2)
#             plot(
#                 f"src_image" + text,
#                 src_image,
#                 src_bbox, 
#                 src_labels if 'src_labels' in locals() else None, 
#                 src_kps if 'src_kps' in locals() else None
#             )
        
# #---------------------------------------------------------------------------
# #
# #---------------------------------------------------------------------------
# def Vis_PoseSampleTopdownAffine(
#     samples : Union[Dict, List],   
#     batch_id : int =0,
# ):
    
#     def plot(
#         title, image, kpts, bbox=None
#     ):  
#         plt.title(
#             title, fontsize = 12
#         )
#         plt.imshow(image)
#         for i in range(0,kpts.shape[0]):
#             kpt_x = int((kpts[i,0]))
#             kpt_y = int((kpts[i,1]))
#             plt.scatter(kpt_x,kpt_y)
#         if bbox is not None :
#             x1, y1, w, h = bbox
#             ax = plt.gca()
#             patch = plt.Rectangle(
#                 [x1, y1], w, h, 
#                 fill=False, 
#                 edgecolor=[0, 0, 1], 
#                 linewidth=2
#             )
#             ax.add_patch(patch)

#     if not isinstance(samples, (dict, list)):
#         raise TypeError(
#             "input samples must be dict or List"
#         )

#     if isinstance(samples, dict):
#        samples = [samples]

#     if samples[0]['bbox_format'][0] != 'xywh':
#         raise ValueError(
#             "only support samples['bbox_format'] == 'xywh'"
#         )    

#     print(samples[0].keys())
#     PackInputTensorTypeSpec(samples[0],{}, show_log=True)
#     print("\n\n\n")
#     for features in samples :
#         resized_image = features["image"][batch_id].numpy()
#         kpts = features['kps'][batch_id]
#         print(kpts)

        
#         if  features.get('meta_info',None) is not None :
#             meta_info = features['meta_info']
#             image_id = meta_info['image_id'][batch_id]
#             instance_id = meta_info['id'][batch_id]
#             src_image = meta_info['src_image'][batch_id].numpy()
#             src_bbox = meta_info['src_bbox'][batch_id] 

#             src_kpts = meta_info['src_keypoints'][batch_id]
#             src_kpts = tf.reshape(src_kpts,[*(kpts.shape)])
#             #print(src_kpts.shape)
            
#         sub_plots = 2 if 'meta_info' in locals() else 1   
#         plt.figure(figsize=(15, 8))
#         plt.subplot(1,sub_plots, 1)
#         if 'meta_info' in locals():
#             extra_text = f"\n( imgae_id : {image_id}, id : {instance_id} )"
        
#         plot(
#             f"transform_image  -- " + extra_text if 'meta_info' in locals() else "",
#             resized_image,
#             kpts = kpts,
#             bbox = None
#         )
#         if 'meta_info' not in locals():
#             return
#         plt.subplot(1,sub_plots, 2)
#         plot(
#             f"src_image  -- " + extra_text,
#             src_image,
#             kpts = src_kpts,
#             bbox = src_bbox
#         )
