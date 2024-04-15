
from typing import Dict, List, Optional, Tuple, Union, Sequence
from tensorflow import Tensor
import matplotlib.pyplot as plt
import tensorflow as tf
from lib.Registers import CODECS
from .base_codec import BaseCodec
from lib.datasets.transforms.utils import PackInputTensorTypeSpec
from lib.visualization.base_vis import BasePoseVisFun
#---------------------------------------------------------------------
#
#----------------------------------------------------------------------
@CODECS.register_module()

class MegviiHeatmapCodec(BaseCodec):
    VERSION = '1.1.0'
    ENCODER_USE_PRED = False
    ENCODER_GEN_SAMPLE_WEIGHT = True
    GAUSSIAN_KERNEL = {
                    1 : [1.],
                    3 : [.25, 0.5, 0.25], 
                    5 : [0.0625, 0.25, 0.375, 0.25, 0.0625],
                    7 : [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]
    }
    r""" MegviiHeatmapCodec
    Date : 2024/2/20
    Author : Dr. David

  
    
    """
    def __init__(self, 
            use_udp : bool = True,
            num_kps : int =17, 
            kernel_sizes : Union[List[int], int] = [7, 9, 11, 15],
            target_size_xy : Tuple[int, int] = (192,256), 
            heatmap_size_xy : Tuple[int, int] = (48, 64),
            stack_multi_hm : bool = True,
            hm_thr : Union[tf.Tensor, List[float]] = None,
            nms_kernel : int = 5,
            **kwargs
    ):
        super().__init__(**kwargs)

        if isinstance(kernel_sizes, (list, tuple)) :
            self.kernel_sizes = kernel_sizes
        else :
            self.kernel_sizes = [kernel_sizes]
        
        if not all([ isinstance(size, int) and size%2== 1 for size in self.kernel_sizes]) :
            raise TypeError( 
                "all items in kernel_sizes must be 'int' type and odd number \n"
                f"but got kernel_size : {self.kernel_sizes} @{self.__class__.__name__}"
            )


        self.use_udp = use_udp
        self.num_kps = num_kps
        self.num_units = len(kernel_sizes)
 
        self.heatmap_size_xy = heatmap_size_xy
        self.target_size_xy = target_size_xy 
        #self.kernel_sizes = tf.cast(kernel_sizes, dtype=tf.int32
        self.stack_outputs = stack_multi_hm
 
        
        self.GaussianKernels_list = [self.getGaussianKernel(size) for size in self.kernel_sizes]  

        'enable UDP or no disable'
        if use_udp :
            self.feat_stride = ( 
                tf.cast(target_size_xy,dtype=self.compute_dtype)-1. ) / ( tf.cast(heatmap_size_xy,dtype=self.compute_dtype)-1.
            )
        else:
            self.feat_stride = ( 
                tf.cast(target_size_xy,dtype=self.compute_dtype)) / ( tf.cast(heatmap_size_xy,dtype=self.compute_dtype)
            )
        
        'set hm_thr /nms_kernel  to decode'
        self.nms_kernel = nms_kernel
        if isinstance(hm_thr, (Sequence, tf.Tensor)):
            self.hm_thr = tf.constant(hm_thr, dtype=self.compute_dtype)
            assert  self.hm_thr.shape[0] == num_kps, \
             f"hm_thr.shape[0] must be equal to {num_kps}, but got {self.hm_thr.shape[0]} @MSRAHeatmap"  
        else:
            self.hm_thr = tf.zeros(shape=(num_kps,), dtype=self.compute_dtype)  

       
       

    def getGaussianKernel(self, kernel_szie):

        if self.GAUSSIAN_KERNEL.get(kernel_szie,None) is not None:
            return  tf.cast( self.GAUSSIAN_KERNEL[kernel_szie], dtype=self.compute_dtype)
        
        sigma = 0.3*( 
            ( tf.cast(kernel_szie-1, dtype=self.compute_dtype) )*0.5 - 1
        ) + 0.8

        x = tf.cast(
            tf.range(-kernel_szie // 2 + 1, kernel_szie // 2 + 1),
            dtype=self.compute_dtype
        )  
        blur_filter = tf.exp(
                -tf.pow(x, 2.0)
                / (2.0 * tf.pow(tf.cast(sigma, dtype=self.compute_dtype), 2.0))
        ) 
        blur_filter /= tf.reduce_sum(blur_filter)
        #return tf.cast( self.GAUSSIAN_KERNEL[3], dtype=self.compute_dtype)
        return tf.cast( blur_filter, dtype=self.compute_dtype)
        
       

    def GaussianBlur(self, heatmaps, kernel):
        size = kernel.shape[0]
        blur_v = tf.reshape(kernel, [size, 1, 1, 1])
        blur_h = tf.reshape(kernel, [1, size, 1, 1])
    
        blur_h = tf.cast(
            tf.tile(blur_h, [1, 1, self.num_kps, 1]), dtype=self.compute_dtype
        )
        blur_v = tf.cast(
            tf.tile(blur_v, [1, 1, self.num_kps, 1]), dtype=self.compute_dtype
        )
        blur_heatmaps = tf.nn.depthwise_conv2d(
            heatmaps, blur_h, strides=[1, 1, 1, 1], padding="SAME"
        )
        blur_heatmaps = tf.nn.depthwise_conv2d(
            blur_heatmaps, blur_v, strides=[1, 1, 1, 1], padding="SAME"
        )
        return blur_heatmaps
    
    def gen_targets(self, kps_true):

        'convert kps to heatmap coordinate and cast as dtype=int'
        kps_hm_xy = kps_true[...,:2]/self.feat_stride #(b,17,,2)
        kps_hm_xy = tf.cast(kps_hm_xy , dtype=tf.int32) #(b,17,,2) @int
        vis = tf.cast(kps_true[...,2] , dtype=tf.int32) #(b,17)    @int

        'condition of kps on heatmap coordinate'
        cond_max_val = tf.greater_equal(kps_hm_xy, self.heatmap_size_xy) #(b,17,,2)
        cond_min_val = tf.less( kps_hm_xy, 0) #(b,17,2)
        cond_vis = tf.equal(vis[...,None], 0) #(b,17,1)
        cond = tf.concat([cond_max_val,cond_min_val, cond_vis], axis=-1) #(b,17,5)
        cond = tf.math.reduce_any(cond, axis=-1, keepdims=True) #(b,17,1)
        kps_hm_true = tf.where(cond,[0,0], kps_hm_xy) #(b,17,2)                  
        kps_hm_true = tf.concat(
            [kps_hm_true, tf.cast(tf.math.logical_not(cond), dtype=tf.int32)] , axis=-1
        ) #(b,17,3) @int


        'condition of kps on heatmap coordinate'
        indices_batch_joints = tf.cast( 
            tf.where( tf.greater(kps_hm_true[...,2], 0) ), dtype=tf.int32   
        )  #(None, 2=[batch_idx, joint_idx]) @int

        eff_kps_hm = tf.boolean_mask(kps_hm_true[...,:2], tf.greater(kps_hm_true[:,:,2],0)) #(None,2) ,xy
        eff_kps_hm = tf.reverse(eff_kps_hm, axis=[-1])  #(None,2) , xy->yx

        indices = tf.concat([indices_batch_joints,eff_kps_hm],axis=-1) # (None,4=[batch_idx, joint_idx, kps_y, kps_x]) 

        heatmap = tf.scatter_nd(
                indices = indices, 
                updates=tf.ones_like(indices[:,0], dtype=self.compute_dtype), 
                shape=[tf.shape(kps_true)[0], self.num_kps, self.heatmap_size_xy[1], self.heatmap_size_xy[0]]
        ) #(b,17,h,w)
        heatmap = tf.transpose(heatmap,(0,2,3,1)) #(b,h,w,17)

        multi_heatmaps = []
        for i in range (self.num_units):
            blur_heatmap = self.GaussianBlur(heatmap, self.GaussianKernels_list[i]) #(b, h, w, 17)
            max_val = tf.reduce_max(blur_heatmap, axis=[1,2]) #(b, 17)
            norm = tf.where(tf.not_equal(max_val, 0.), max_val, 1.) #(b, 17)
            blur_heatmap = blur_heatmap/norm[:, None,None,:]*255
            multi_heatmaps.append(blur_heatmap) #(b,h,w,17)

        if self.num_units > 1 :
            multi_heatmaps = tf.stack(multi_heatmaps, axis=1)
        else:
            multi_heatmaps = multi_heatmaps[0]

        sample_weights =  tf.where(
            cond,
            tf.cast([0.,0.,0.], dtype=self.compute_dtype), 
            kps_true
        ) #(b,17,2)
        #sample_weights =  tf.tile(sample_weights[:,None,...], (1, self.num_units,1,1)) 

        if  self.embedded_codec :
            sample_weights = tf.stop_gradient(sample_weights)
            multi_heatmaps = tf.stop_gradient(multi_heatmaps)

        return multi_heatmaps, sample_weights


    def batch_encode(
            self,
            data : Dict[str,tf.Tensor],
            y_pred : Optional[ Union[Tuple[Tensor],Tensor]]=None
    ) -> Dict[str,tf.Tensor]:
        
        " y_pred not used"
        #kps_true = data['kps']
        multi_heatmaps, sample_weights = self.gen_targets( 
            tf.cast(data['kps'], dtype=self.compute_dtype)
        )
        if not self.stack_outputs and self.num_units > 1:
            multi_heatmaps = [
                tf.squeeze(feat, axis=1) for feat in tf.split(multi_heatmaps, self.num_units, axis=1)
            ]
            # sample_weights = [tf.squeeze(feat, axis=1) 
            #                   for feat in tf.split(sample_weights, self.num_units, axis=1)
            # ]

        data['y_true'] = multi_heatmaps
        data['sample_weight'] = sample_weights
        return data
    
    def batch_decode(
            self, 
            data : Dict[str,Union[List[Tensor],Tensor]], 
            meta_data : Optional[dict] = None,
            *args, **kwargs
    ) -> tf.Tensor:
        r"""
        to_src_image = dict(
            aspect_scale_xy = tf.constant([1.,1.], dtype=self.compute_dtype),
            offset_padding_xy = tf.constant([0.,0.], dtype=self.compute_dtype),
            bbox_lt_xy= tf.constant([0.,0.], dtype=self.compute_dtype)
        )
        
        """
        #y_pred = data['y_pred']
        if isinstance(data['y_pred'], Sequence):
            #multi_stage , get y_pred of the final stage
            y_pred = data['y_pred'][-1]
        else:
            #single_stage
            y_pred = data['y_pred'] 


        'y_pred (batch_multi_heatmaps) : (b,num_units,64,48,joints)'
        if y_pred.shape.rank == 4:
           y_pred = tf.expand_dims(y_pred, axis=0) 
        elif  y_pred.shape.rank == 5:
            pass
        else:
            raise ValueError(

            )
        
        batch_heatmaps = y_pred[:, 0, ...] #only decode preds of P2-feat
        _, height, width, joints = batch_heatmaps.shape
    
        assert height==self.heatmap_size_xy[1] and width==self.heatmap_size_xy[0], \
        f"heatmaps_wh to decode is {width}x{height}, it didn't match {self.heatmap_size_xy} @MSRAHeatmap.batch_decode"
        
        assert ( joints==self.num_kps),f"joints_num must be {self.num_kps}, but got {joints} @MSRAHeatmap.batch_decode"

        def nms(heat, kernel=5):
            hmax = tf.nn.max_pool2d(heat, kernel, 1, padding='SAME')
            keep = tf.cast(tf.equal(heat, hmax), self.compute_dtype)
            return heat*keep
        
        batch_heatmaps = nms(batch_heatmaps, kernel=self.nms_kernel ) #(1,96,96,14)
        flat_tensor = tf.reshape(batch_heatmaps, [tf.shape(batch_heatmaps)[0], -1, tf.shape(batch_heatmaps)[-1]]) #(1,96*96,14)
        'Argmax of the flat tensor'
        argmax = tf.argmax(flat_tensor, axis=1) #(1,14)
        argmax = tf.cast(argmax, tf.int32) #(1,14)
        scores = tf.math.reduce_max(flat_tensor, axis=1) #(1,14)
        'Convert indexes into 2D coordinates'
        argmax_y = tf.cast( (argmax//width), self.compute_dtype)  #(1,14)
        argmax_x = tf.cast( (argmax%width), self.compute_dtype)   #(1,14)

        # Shape: batch * 3 * n_points  (1,14,3)
        batch_keypoints = tf.stack((argmax_x, argmax_y, scores), axis=2)
        'mask'
        mask = tf.greater(batch_keypoints[..., 2], self.hm_thr[None,:]) #(b,17)
        kps_xy = tf.where(mask[:,:,None], batch_keypoints[:, :, :2], 0)*self.feat_stride
        batch_keypoints = tf.concat([kps_xy, scores[:,:,None]], axis=-1) 


        if meta_data:
            'get kps of src frame'
            kps_xy = self.transform_preds(
                batch_keypoints[...,:2], 
                mask, 
                meta_data
            )
            'concat src_ksp and socre of heatmap'
            batch_keypoints = tf.concat([kps_xy, batch_keypoints[:,:,2:]], axis=-1) 

        # if normalize:
        #     argmax_x = argmax_x / tf.cast(width, self.compute_dtype)
        #     argmax_y = argmax_y / tf.cast(height, self.compute_dtype)
        # if expand_batch_dim:
        # return tf.squeeze(batch_keypoints, axis=0)  
            
        data['decode_pred'] = batch_keypoints
        return data 
    
    def vis_encoder_res(
            self, 
            tfrec_datasets_list : Union[List, Dict], 
            transforms_list  : Optional[Union[List, Dict]]=None, 
            show_decoder_res : bool = True,
            take_batch_num : int = 1,
            vis_fn : Optional[callable] =  None,
            batch_ids :  Union[List[int], int] = [0], 
            figsize : Tuple[int] = (36, 10), 
            
        ):

        if vis_fn is None :
            vis_fn =  Vis_TopdownPoseMultiHeatmapsCodec(
                figsize =figsize,
                sel_batch_ids =batch_ids,
                num_heatmap_units = self.num_units 
            )
    
        else:
            if not isinstance(vis_fn,BasePoseVisFun):
                raise TypeError(
                    f"vis_fn must be 'BasePoseVisFun' type {self.__class__.__name__}"
                )

        #Vis_PoseSampleTopdownAffine
    
        if transforms_list is None :
            from lib.datasets.transforms import TopdownAffine
            img_affine = TopdownAffine(
                is_train = True, 
                test_mode = False,   
                do_clip = True,      
                keep_bbox_aspect_prob = 0.5,
                rotate_prob = 0.5,
                shear_prob = 0.,
                MaxRot_deg=30., 
                MaxShear_deg=15.
            )
            transforms_list = [img_affine]

        batch_dataset = super().gen_tfds_w_codec(
            tfrec_datasets_list = tfrec_datasets_list, 
            transforms_list = transforms_list, 
            test_mode=True, 
            batch_size=16, 
            shuffle=True
        )
        
        batched_samples_list = [ feats for feats in batch_dataset.take(take_batch_num)]
        batched_decode_pred_list = [self(sample["y_true"], codec_type='decode') for sample in batched_samples_list] if show_decoder_res else None

        vis_fn(
            batched_samples_list,    
            decode_pred = batched_decode_pred_list,
        )

        del batch_dataset

#---------------------------------------------------------------------
#
#----------------------------------------------------------------------
class Vis_TopdownPoseMultiHeatmapsCodec(BasePoseVisFun):
    VERSION = '1.0.0'

    R"""  Vis_TopdownPoseSimCCLabelCodec
    Author : Dr. David Shen
    Date : 2024/3/29
    i.e. :

    """
    def __init__(
            self,
            num_heatmap_units  : int = 4,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heatmap_units = num_heatmap_units
        
    def parse_dataset(
        self, 
        batched_samples_list : List[dict], 
    ) :

        batched_src_samples_list = [samples.pop('meta_info', None) for samples in batched_samples_list]
        # src_kpts =  self.extract_key_samples(batched_src_samples_list, 'src_keypoints')
        # src_kpts =  [ tf.reshape(kps,[-1,17,3]) for kps in src_kpts] if self.all_vaild(src_kpts) else None

        data_dict = dict(
            img = self.extract_key_samples(batched_samples_list, 'image'),
            encode_true = self.extract_key_samples(batched_samples_list, 'y_true'),
            kpts = self.extract_key_samples(batched_samples_list, 'kps'),
            y_pred =  self.extract_key_samples(batched_samples_list, 'y_pred'),
            decode_pred =  self.extract_key_samples(batched_samples_list, 'decode_pred'),
            src_img = self.extract_key_samples(batched_src_samples_list, 'src_image'),
            src_bboxes = self.extract_key_samples(batched_src_samples_list, 'src_bbox'),
            src_img_id = self.extract_key_samples(batched_src_samples_list, 'image_id'),
            src_ids = self.extract_key_samples(batched_src_samples_list, 'id'),
            src_kpts = self.extract_key_samples(batched_src_samples_list, 'src_keypoints'),

        )
        del batched_samples_list
        return data_dict

    def call(
        self, 
        img : Tensor,
        y_pred : Optional[Tensor] = None,
        decode_pred : Optional[Tensor] = None,
        encode_true : Optional[Tensor] = None,
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
        np_resized_img =  None if img  is None else img.numpy()
        np_src_img = None if src_img is None else src_img.numpy()  
        image_id = '' if src_img_id is None else src_img_id
        instance_id = '' if src_ids is None else src_ids
        
        'encode data'
        plot_hm_true = True if encode_true is not None  else False
        plot_kpt_true = True  if np_resized_img is not None else False 
        plot_hm_pred = True if y_pred is not None  else False
        plot_kpt_pred = True  if decode_pred is not None else False 
        plot_src_img = True if np_src_img is not None else False

        if plot_hm_true : 
            multi_heatmaps = encode_true
        if plot_hm_pred :
            multi_heatmaps = y_pred

        base_sub_plots = sum(
            [(plot_hm_true or plot_hm_pred)*self.num_heatmap_units,plot_kpt_true,plot_kpt_pred, plot_src_img*2]
        )



        plt.figure(figsize=self.figsize)
        idx = 0
        '#1 resized img with gt_kpts'
        if plot_kpt_true :
            idx += 1
            self.base_plot(
                title = f"Resized Image with gt-kpts",
                image = np_resized_img,
                kpts = kpts,
                bboxes = None,
                subplot_spec = (1,base_sub_plots,idx) 
            )

        '#2 gt_heatmaps'
        if plot_hm_true or  plot_hm_pred:
            for unit_id in range (self.num_heatmap_units):
                idx += 1
                title = f"heatmaps_{'true' if plot_hm_true else 'pred'} sum "
                self.heatmaps_plot(
                    y_trues = multi_heatmaps[unit_id,...], 
                    sum_axis= -1, 
                    title = title + f' (feat :  P{unit_id+2})', 
                    subplot_spec=(1,base_sub_plots,idx)          
                ) 

        '#3 resized img with pred_kpts'
        if plot_kpt_pred :
            idx += 1
            self.base_plot(
                title = f"Resized Image with pred_kpts",
                image = np_resized_img,
                kpts = kpts,
                bboxes = None,
                subplot_spec = (1,base_sub_plots,idx)
            )

        '#4 src_img'         
        if plot_src_img:
            idx += 1
            extra_text = f"\n( imgae_id : {image_id}, id : {instance_id} )"
            self.base_plot(
                title = f"Src Image" + extra_text,
                image = np_src_img,
                kpts = src_kpts,
                bboxes = src_bboxes,
                subplot_spec = (1,4,4)

            )   
# #---------------------------------------------------------------------
# #
# #----------------------------------------------------------------------
# def Vis_TopdownPoseMultiHeatmapsCodec(
#     batched_samples : Union[Dict, List],   
#     batch_ids : int =0,
#     batched_decoded_kpts : Optional[Union[tf.Tensor, List]] = None,
#     show_kpts : bool = True
# ):
#     import matplotlib.pyplot as plt 
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

#     if not isinstance(batched_samples, (dict, list)):
#         raise TypeError(
#             "input samples must be dict or List"
#         )

#     if isinstance(batched_samples, dict):
#        batched_samples = [batched_samples]

#     if batched_decoded_kpts is not None :
#         if not isinstance(batched_samples, (tf.Tensor, list)):
#             raise TypeError(
#                 "type of batched_decoded_kpts must be 'tf.Tensor' or 'List[Tensor]'"
#             )
#         if isinstance(batched_decoded_kpts, tf.Tensor):
#             batched_decoded_kpts = [batched_decoded_kpts] 

    
#     if batched_samples[0]['bbox_format'][0] != 'xywh':
#         raise ValueError(
#             "only support samples['bbox_format'] == 'xywh'"
#         )  
  

#     print(batched_samples[0].keys())
#     PackInputTensorTypeSpec(batched_samples[0],{}, show_log=True)
#     print("\n\n\n")

#     base_sub_plots = 1 
#     for ith, features in enumerate(batched_samples):
#         for batch_id in batch_ids :
#             sub_plots = base_sub_plots
            
#             resized_image = features["image"][batch_id].numpy()
#             kpts = features['kps'][batch_id]
#             multi_heatmaps = features['y_true'][batch_id].numpy()
#             num_units = multi_heatmaps.shape[0]
#             sub_plots += num_units

#             if show_kpts :
#                 print(kpts)
#             if batched_decoded_kpts is not None:
#                decoded_kpts = batched_decoded_kpts[ith][batch_id] 
#                sub_plots += 1 
            
#             #print(kpts)
#             if  features.get('meta_info',None) is not None :
#                 meta_info = features['meta_info']
#                 image_id = meta_info['image_id'][batch_id]
#                 instance_id = meta_info['id'][batch_id]
#                 src_image = meta_info['src_image'][batch_id].numpy()
#                 src_bbox = meta_info['src_bbox'][batch_id] 
#                 src_kpts = meta_info['src_keypoints'][batch_id]
#                 src_kpts = tf.reshape(src_kpts,[*(kpts.shape)])
#                 sub_plots += 1 
#             #sub_plots = 4 if 'meta_info' in locals() else 3 
#             #if 'meta_info' in locals() :  sub_plots += 1 
            

#             if 'meta_info' in locals():
#                 sub_plots += 1 
#                 extra_text = f"\n( imgae_id : {image_id}, id : {instance_id} )"

#             plt.figure(figsize=(36, 10))
#             plt.subplot(1,sub_plots, 1)
#             plot(
#                 f"transform_image  " + extra_text if 'meta_info' in locals() else "",
#                 resized_image,
#                 kpts = kpts,
#                 bbox = None
#             )

#             'heatmap'
#             #multi_heatmaps = features['y_true'][batch_id].numpy()
#             num_units = multi_heatmaps.shape[0]
#             for unit_id in range (num_units):
#                 plt.subplot(1, sub_plots, unit_id+2)
#                 plt.title(f'kps_heatmap sum (feat :  P{unit_id+2})',fontsize= 12)
#                 heatmaps_plot = multi_heatmaps[unit_id, :, :, :]
#                 plt.imshow(heatmaps_plot.sum(axis=-1))

#             'kps from heatmap by decoder'
#             if 'decoded_kpts' in locals():
#                 plt.subplot(1,sub_plots, num_units+2)
#                 plot(
#                     f"transform_image with kps from decoder(P2)" + extra_text if 'meta_info' in locals() else "",
#                     resized_image,
#                     kpts = decoded_kpts,
#                     bbox = None
#                 )     

#             if 'meta_info' not in locals():
#                 return
#             plt.subplot(1,4, 4)
#             plot(
#                 f"src_image  " + extra_text,
#                 src_image,
#                 kpts = src_kpts,
#                 bbox = src_bbox
#             )






# class MegviiHeatmapCodec(BaseCodec):
#     VERSION = '1.0.0'
#     ENCODER_USE_PRED = False
#     ENCODER_GEN_SAMPLE_WEIGHT = True
#     GAUSSIAN_KERNEL = {
#                     1 : [1.],
#                     3 : [.25, 0.5, 0.25], 
#                     5 : [0.0625, 0.25, 0.375, 0.25, 0.0625],
#                     7 : [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]
#     }
#     r""" MegviiHeatmapCodec

  
    







#     """
#     def __init__(self, 
#             use_udp : bool = True,
#             num_kps : int =17, 
#             kernel_sizes : Union[List[int], int] = [7, 9, 11, 15],
#             target_size_xy : Tuple[int, int] = (192,256), 
#             heatmap_size_xy : Tuple[int, int] = (48, 64),
#             stack_multi_hm : bool = True,
#             hm_thr : Union[tf.Tensor, List[float]] = None,
#             **kwargs
#     ):
#         super().__init__(**kwargs)

#         if isinstance(kernel_sizes, (list, tuple)) :
#             self.kernel_sizes = kernel_sizes
#         else :
#             self.kernel_sizes = [kernel_sizes]
        
#         if not all([ isinstance(size, int) and size%2== 1 for size in self.kernel_sizes]) :
#             raise TypeError( 
#                 "all items in kernel_sizes must be 'int' type and odd number \n"
#                 f"but got kernel_size : {self.kernel_sizes} @{self.__class__.__name__}"
#             )


#         self.use_udp = use_udp
#         self.num_kps = num_kps
#         self.num_units = len(kernel_sizes)

#         self.heatmap_size_xy = heatmap_size_xy
#         self.target_size_xy = target_size_xy 
#         #self.kernel_sizes = tf.cast(kernel_sizes, dtype=tf.int32)
#         #self.kernel_sizes = kernel_sizes
#         self.stack_outputs = stack_multi_hm

#         'enable UDP or no disable'
#         if use_udp :
#             self.feat_stride = ( tf.cast(target_size_xy,dtype=self.compute_dtype)-1. ) / ( tf.cast(heatmap_size_xy,dtype=self.compute_dtype)-1.)
#         else:
#             self.feat_stride = ( tf.cast(target_size_xy,dtype=self.compute_dtype)) / ( tf.cast(heatmap_size_xy,dtype=self.compute_dtype))
        
#         'set hm_thr to decode'
#         if isinstance(hm_thr, (Sequence, tf.Tensor)):
#             self.hm_thr = tf.constant(hm_thr, dtype=self.compute_dtype)
#             assert  self.hm_thr.shape[0] == num_kps, \
#              f"hm_thr.shape[0] must be equal to {num_kps}, but got {self.hm_thr.shape[0]} @MSRAHeatmap"  
#         else:
#             self.hm_thr = tf.zeros(shape=(num_kps,), dtype=self.compute_dtype)  

#         self.GaussianKernels_list = [self.getGaussianKernel(size) for size in self.kernel_sizes]  
       

#     def getGaussianKernel(self, kernel_szie):

#         if self.GAUSSIAN_KERNEL.get(kernel_szie,None) is not None:
#             return  tf.cast( self.GAUSSIAN_KERNEL[kernel_szie], dtype=self.compute_dtype)
        
#         sigma = 0.3*( 
#             ( tf.cast(kernel_szie-1, dtype=self.compute_dtype) )*0.5 - 1
#         ) + 0.8

#         x = tf.cast(
#             tf.range(-kernel_szie // 2 + 1, kernel_szie // 2 + 1),
#             dtype=self.compute_dtype
#         )  
#         blur_filter = tf.exp(
#                 -tf.pow(x, 2.0)
#                 / (2.0 * tf.pow(tf.cast(sigma, dtype=self.compute_dtype), 2.0))
#         ) 
#         blur_filter /= tf.reduce_sum(blur_filter)
#         #return tf.cast( self.GAUSSIAN_KERNEL[3], dtype=self.compute_dtype)
#         return tf.cast( blur_filter, dtype=self.compute_dtype)
        
       

#     def GaussianBlur(self, heatmaps, kernel):
#         size = kernel.shape[0]
#         blur_v = tf.reshape(kernel, [size, 1, 1, 1])
#         blur_h = tf.reshape(kernel, [1, size, 1, 1])
    
#         blur_h = tf.cast(
#             tf.tile(blur_h, [1, 1, self.num_kps, 1]), dtype=self.compute_dtype
#         )
#         blur_v = tf.cast(
#             tf.tile(blur_v, [1, 1, self.num_kps, 1]), dtype=self.compute_dtype
#         )
#         blur_heatmaps = tf.nn.depthwise_conv2d(
#             heatmaps, blur_h, strides=[1, 1, 1, 1], padding="SAME"
#         )
#         blur_heatmaps = tf.nn.depthwise_conv2d(
#             blur_heatmaps, blur_v, strides=[1, 1, 1, 1], padding="SAME"
#         )
#         return blur_heatmaps
    
#     def gen_targets(self, kps_true):

#         'convert kps to heatmap coordinate and cast as dtype=int'
#         kps_hm_xy = kps_true[...,:2]/self.feat_stride #(b,17,,2)
#         kps_hm_xy = tf.cast(kps_hm_xy , dtype=tf.int32) #(b,17,,2) @int
#         vis = tf.cast(kps_true[...,2] , dtype=tf.int32) #(b,17)    @int

#         'condition of kps on heatmap coordinate'
#         cond_max_val = tf.greater_equal(kps_hm_xy, self.heatmap_size_xy) #(b,17,,2)
#         cond_min_val = tf.less( kps_hm_xy, 0) #(b,17,2)
#         cond_vis = tf.equal(vis[...,None], 0) #(b,17,1)
#         cond = tf.concat([cond_max_val,cond_min_val, cond_vis], axis=-1) #(b,17,5)
#         cond = tf.math.reduce_any(cond, axis=-1, keepdims=True) #(b,17,1)
#         kps_hm_true = tf.where(cond,[0,0], kps_hm_xy) #(b,17,2)                  
#         kps_hm_true = tf.concat(
#             [kps_hm_true, tf.cast(tf.math.logical_not(cond), dtype=tf.int32)] , axis=-1
#         ) #(b,17,3) @int


#         'condition of kps on heatmap coordinate'
#         indices_batch_joints = tf.cast( 
#             tf.where( tf.greater(kps_hm_true[...,2], 0) ), dtype=tf.int32   
#         )  #(None, 2=[batch_idx, joint_idx]) @int

#         eff_kps_hm = tf.boolean_mask(kps_hm_true[...,:2], tf.greater(kps_hm_true[:,:,2],0)) #(None,2) ,xy
#         eff_kps_hm = tf.reverse(eff_kps_hm, axis=[-1])  #(None,2) , xy->yx

#         indices = tf.concat([indices_batch_joints,eff_kps_hm],axis=-1) # (None,4=[batch_idx, joint_idx, kps_y, kps_x]) 

#         heatmap = tf.scatter_nd(
#                 indices = indices, 
#                 updates=tf.ones_like(indices[:,0], dtype=self.compute_dtype), 
#                 shape=[tf.shape(kps_true)[0], self.num_kps, self.heatmap_size_xy[1], self.heatmap_size_xy[0]]
#         ) #(b,17,h,w)
#         heatmap = tf.transpose(heatmap,(0,2,3,1)) #(b,h,w,17)

#         multi_heatmaps = []
#         for i in range (self.num_units):
#             blur_heatmap = self.GaussianBlur(heatmap, self.GaussianKernels_list[i]) #(b, h, w, 17)
#             max_val = tf.reduce_max(blur_heatmap, axis=[1,2]) #(b, 17)
#             norm = tf.where(tf.not_equal(max_val, 0.), max_val, 1.) #(b, 17)
#             blur_heatmap = blur_heatmap/norm[:, None,None,:]*255
#             multi_heatmaps.append(blur_heatmap) #(b,h,w,17)

#         if self.num_units > 1 :
#             multi_heatmaps = tf.stack(multi_heatmaps, axis=1)
#         else:
#             multi_heatmaps = multi_heatmaps[0]

#         sample_weights =  tf.where(cond,[0.,0.,0.], kps_true) #(b,17,2)
#         #sample_weights =  tf.tile(sample_weights[:,None,...], (1, self.num_units,1,1)) 

#         if  self.embedded_codec :
#             sample_weights = tf.stop_gradient(sample_weights)
#             multi_heatmaps = tf.stop_gradient(multi_heatmaps)

#         return multi_heatmaps, sample_weights


#     def batch_encode(self,
#                     data : Dict[str,tf.Tensor],
#                     y_pred : Optional[ Union[Tuple[Tensor],Tensor]]=None) -> Dict[str,tf.Tensor]:
#         " y_pred not used"
        
#         kps_true = data['kps']
#         multi_heatmaps, sample_weights = self.gen_targets(kps_true)
#         if not self.stack_outputs and self.num_units > 1:
#             multi_heatmaps = [tf.squeeze(feat, axis=1) 
#                               for feat in tf.split(multi_heatmaps, self.num_units, axis=1)
#             ]
#             # sample_weights = [tf.squeeze(feat, axis=1) 
#             #                   for feat in tf.split(sample_weights, self.num_units, axis=1)
#             # ]

#         data['y_true'] = multi_heatmaps
#         data['sample_weight'] = sample_weights
#         return data
    
