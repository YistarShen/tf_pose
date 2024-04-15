from typing import Dict, List, Optional, Tuple, Union, Any, Sequence
import math
from tensorflow import Tensor
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .base_codec import BaseCodec
from lib.Registers import CODECS
from lib.visualization.base_vis import BasePoseVisFun




#---------------------------------------------------------------------
#
#----------------------------------------------------------------------
@CODECS.register_module()
class SimCCLabelCodec(BaseCodec):
    VERSION = '1..0'
    ENCODER_USE_PRED = False
    ENCODER_GEN_SAMPLE_WEIGHT = True
    r""" SimCCLabelCodec   Simple Coordinate Classification (simcc)
    Author : Dr. David Shen
    Date : 2024/2/22

    
    """
    def __init__(
        self, 
        sigma_xy =(4.9,5.66), 
        image_size_xy=(192,256), 
        simcc_split_ratio=2.,
        normalize = False,
        use_udp = True, 
        hm_thr : Union[tf.Tensor, List[float]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.image_size_xy = image_size_xy
        self.simcc_ratio = simcc_split_ratio
        self.target_size_xy = (int(image_size_xy[0]*self.simcc_ratio),int(image_size_xy[1]*self.simcc_ratio) )
        self.normalize = normalize 
    
        self.sigma_xy = tf.constant(sigma_xy, dtype=self.compute_dtype) #(2,)
        self.sigma_square_xy = tf.math.square(self.sigma_xy)  #(2,)
        self.radius_xy = self.sigma_xy*3. #(2,)

        if self.normalize :
            self.sigma_sqrt_2pi = self.sigma_xy*tf.math.sqrt(2.*math.pi) #(2,)

        'enable UDP or no disable'
        if use_udp :
            self.simcc_split_ratio =(
                tf.cast(self.target_size_xy, dtype=self.compute_dtype) -1.)/(tf.cast(self.image_size_xy,dtype=self.compute_dtype)-1.
            )  
        else:
            self.simcc_split_ratio =(
                tf.cast(self.target_size_xy, dtype=self.compute_dtype))/(tf.cast(self.image_size_xy,dtype=self.compute_dtype)
            )
        'set hm_thr to decode'
        if isinstance(hm_thr, (Sequence, tf.Tensor)):
            self.hm_thr = tf.constant(hm_thr, dtype=self.compute_dtype)

        print(f"target_size_xy : {self.target_size_xy}")
        print(f"sigma_xy : {self.sigma_xy}")
        print(f"simdr_split_ratio : {self.simcc_split_ratio}")

        self.simcc_coords_x = tf.cast(
            tf.range(0, self.target_size_xy[0]), dtype=self.compute_dtype
        )
        self.simcc_coords_y = tf.cast(
            tf.range(0, self.target_size_xy[1]), dtype=self.compute_dtype
        )

        'kps confidence threhold config'
        #self.val_thr = tf.ones(shape=(17,1), dtype=self.compute_dtype)*0.0

    def gen_simcc_targets(self,kps_true):

        'kps transform to simcc from image size(256x192)'
        mu_xy = kps_true[:,:,:2]*self.simcc_split_ratio #(b,17,2)

        # lu = tf.cast( 
        #     mu_xy[:,:,:2] - self.temp_size_xy, dtype=tf.int32
        # )   #(b,17,2,)
        # rb = tf.cast( 
        #     mu_xy[:,:,:2] + self.temp_size_xy+1, dtype=tf.int32
        # )  #(b,17,2,)
        lu = mu_xy - self.radius_xy #(b,17,2,)
        rb = mu_xy + self.radius_xy+1 #(b,17,2,)

        mask_lu = tf.greater_equal(lu, self.target_size_xy) #(b, 17,2,)
        mask_rb = tf.less(rb, [0,0]) #(b, 17,2,)
        mask_vis = tf.equal(kps_true[:, :, 2:3], 0.)#(b,17,1)
        # mask = tf.reduce_any(tf.concat([mask_lu, mask_rb, mask_vis],axis=-1), axis=-1, keepdims=True) #(b,17,5) => (b,17,1)
        mask = tf.reduce_any(
            tf.concat([mask_lu, mask_rb, mask_vis],axis=-1), axis=-1, keepdims=False
        ) #(b,17,5) => (b,17)
        #target_weights = tf.where(mask,0.,1.) #(b,17)
        target_weights = tf.cast( 
            tf.math.logical_not(mask),dtype=self.compute_dtype
        )#(b,17)

        target_x = tf.math.squared_difference( 
            self.simcc_coords_x[None,None,:], tf.math.round(mu_xy[:,:,0,None])
        ) # (b, 17, None)x(None, None, coords_x)=>(b, 17, coords_x,)
        target_y = tf.math.squared_difference(
            self.simcc_coords_y[None,None,:], tf.math.round(mu_xy[:,:,1,None])
        ) #(b, 17, coords_y,)
        target_x = tf.math.exp( -target_x/(2.*self.sigma_square_xy[0])) #(b,17, coords_x,)
        target_y = tf.math.exp( -target_y/(2.*self.sigma_square_xy[1])) #(b, 17, coords_y,)

        if self.normalize :
            target_x /= self.sigma_sqrt_2pi[0]
            target_y /= self.sigma_sqrt_2pi[1]
    
        target_x = target_x*target_weights[...,None] #(b,17, coords_x)*(b,17, 1)=>(b,17, coords_x)
        target_y = target_y*target_weights[...,None] 
        target_xy = tf.concat([target_x,target_y], axis=-1)  #(b, 17, coords_x+coords_y)
        #sample_weights = tf.concat([kps_true[:, :, :2], target_weights], axis=-1)    

        # if self.embedded_codec:
        #     target_weights = tf.stop_gradient(target_weights)
        #     target_xy = tf.stop_gradient(target_xy)
        return target_xy, target_weights



    def batch_encode(
            self, 
            data : Dict[str,tf.Tensor], 
            y_pred : Optional[tf.Tensor]  = None
    ) -> Dict[str,tf.Tensor]:
        'add new keys for encode results'
        #kps_true = data['kps']
        'kps transform to simcc from image size(256x192)'
        target_xy, sample_weights =  self.gen_simcc_targets(
            tf.cast(data['kps'], dtype=self.compute_dtype)
        )
        data['y_true'] =  target_xy  #simicc--encode : (b, 17, coords_x+coords_y)
        data['sample_weight'] = sample_weights #coordinate based on model's input dimemsion
        return data

    
    def batch_decode(
            self, 
            data : Dict[str,Tensor], 
            val_thr : Optional[Tensor]=None,
            *args, **kwargs
    ) ->Tensor:
        
        y_pred = data['y_pred']
        #y_pred = data

        num_kps = tf.shape(y_pred)[1]
        'kps confidence threhold config'
        if val_thr is None :
            val_thr = getattr(self,'hm_thr', tf.ones(shape=(num_kps,), dtype=self.compute_dtype)*0.0)
        else:
            val_thr = tf.cast(val_thr,dtype=self.compute_dtype)

        # assert  val_thr.shape[0] == num_kps, \
        # f"hm_thr.shape[0] must be equal to {num_kps}, but got {val_thr.shape[0]} @MSRAHeatmap"    
  
        'kps_from_simcc'

        batch_simcc_x, batch_simcc_y = tf.split(y_pred, num_or_size_splits=[*self.target_size_xy], axis=-1) #(b,17,192*2)

        argmax_x = tf.argmax(batch_simcc_x, axis=-1) #(b,17)
        argmax_y = tf.argmax(batch_simcc_y, axis=-1) #(b,17)

        argmax_xy = tf.cast(tf.stack([argmax_x,argmax_y], axis=-1), dtype=self.compute_dtype)/self.simcc_split_ratio  #(b,17,2)
        
        max_val_x  = tf.math.reduce_max(batch_simcc_x, axis=-1)  #(b,17)
        max_val_y  = tf.math.reduce_max(batch_simcc_y, axis=-1)  #(b,17)
        val = tf.math.minimum(max_val_x,max_val_y) #(b,17,)

        mask =  tf.greater(val, val_thr) #(b,17)
        batch_keypoints = tf.where(mask[...,None], argmax_xy, 0.) #(b,17,2)
        batch_keypoints = tf.concat([batch_keypoints, val[...,None]], axis=-1) #(b,17,3)    
        
        data['decode_pred'] = batch_keypoints
        #return batch_keypoints
        return data
       
    # def vis_encoder_res(
    #         self, 
    #         tfrec_datasets_list : Union[List, Dict], 
    #         transforms_list  : Optional[Union[List, Dict]]=None, 
    #         vis_fn : Optional[callable] =  None,
    #         take_batch_num : int = 1,
    #         batch_ids :  Union[List[int], int] = [0],  
    #         show_decoder_res : bool = True
    # ):


    #     if transforms_list is None :
    #         from lib.datasets.transforms import TopdownAffine
    #         img_affine = TopdownAffine(
    #             is_train = True, 
    #             test_mode = False,   
    #             do_clip = True,      
    #             keep_bbox_aspect_prob = 0.5,
    #             rotate_prob = 0.5,
    #             shear_prob = 0.,
    #             MaxRot_deg=30., 
    #             MaxShear_deg=15.
    #         )
    #         transforms_list = [img_affine]

    #     batch_dataset = super().gen_tfds_w_codec(
    #         tfrec_datasets_list = tfrec_datasets_list, 
    #         transforms_list = transforms_list, 
    #         test_mode=True, 
    #         batch_size=16, 
    #         shuffle=True
    #     )


    #     if vis_fn is None :
    #         vis_fn =  Vis_TopdownPoseSimCCLabelCodec(
    #             simicc_split_xy_dim =   self.target_size_xy,
    #             figsize = (24, 8),
    #             sel_batch_ids  = batch_ids
    #         )
            
    #     batched_samples_list = [ feats for feats in batch_dataset.take(take_batch_num)]

    #     batched_images_list =  [ samples['image'] for samples in batched_samples_list]
    #     batched_encode_true_list =  [ samples['y_true'] for samples in batched_samples_list]
    #     batched_decode_pred_list =  [ self.batch_decode(samples['y_true'])  for samples in batched_samples_list]
    #     batched_kpts_list =  [ samples['kps']  for samples in batched_samples_list]


    #     batched_decoded_kpts =  [ self.batch_decode(smaples['y_true']) for smaples in batched_samples] if show_decoder_res else None

    #     vis_fn(
    #         img = batched_images_list,
    #         decode_pred  = batched_decode_pred_list,
    #         encode_true  = encoded_trues_list,
    #         kpts = batched_kpts_list,
    #         src_img  = batched_src_imgs_list,
    #         src_bboxes  = batched_src_bboxes_list,
    #         src_kpts  = batched_src_kpts_list,
    #         src_img_id  = batched_src_img_ids_list,
    #         src_ids = batched_src_ids_list,
    #     )
    #     vis_fn(
    #         batched_samples,    
    #         batch_ids,
    #         batched_decoded_kpts
    #     )
        

    #     del batch_dataset 

    

    def vis_encoder_res(
            self, 
            tfrec_datasets_list : Union[List, Dict], 
            transforms_list  : Optional[Union[List, Dict]]=None, 
            vis_fn : Optional[callable] =  None,
            take_batch_num : int = 1,
            batch_ids :  Union[List[int], int] = [0],  
            show_decoder_res : bool = True
    ):

        if vis_fn is None :
            vis_fn =  Vis_TopdownPoseSimCCLabelCodec(
                figsize = (16, 8),
                sel_batch_ids  = batch_ids,
                plot_transformed_bbox=False,
            )

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
        batched_samples = [ feats for feats in batch_dataset.take(take_batch_num)]
        batched_decoded_kpts =  [ self.batch_decode(smaples['y_true']) for smaples in batched_samples] if show_decoder_res else None
        vis_fn(
            batched_samples_list = batched_samples,    
            decode_pred = batched_decoded_kpts,
            sel_batch_ids = batch_ids
        )
        
        del batch_dataset


    

class Vis_TopdownPoseSimCCLabelCodec(BasePoseVisFun):
    VERSION = '1.0.0'

    R"""  Vis_TopdownPoseSimCCLabelCodec
    
    i.e. :
        vis_fn = Vis_TopdownPoseSimCCLabelCodec(
            simicc_split_xy_dim  = (192*2, 256*2),
            figsize = (24, 8),
            sel_batch_ids  = [0]
        )

        batch_val_dataset = val_tfds_builder.GenerateTargets(
            test_mode=True,     
            unpack_x_y_sample_weight= False, 
        )
        batched_samples_list = [ model.encoder(feat) for feat in batch_val_dataset.take(2)]

        # Method 1 : to list all possiable data for plotting
            encoded_trues_list = [ batched_samples['y_true'] for batched_samples in batched_samples_list]
            batched_images_list = [ batched_samples['image'] for batched_samples in batched_samples_list]
            batched_kpts_list = [ batched_samples['kps'] for batched_samples in batched_samples_list]
            batched_pred_list = [ model(batched_samples['image']) for batched_samples in batched_samples_list]
            batched_decode_pred_list = [ model.decoder(batched_samples) for batched_samples in batched_pred_list]
            batched_src_imgs_list = [ batched_samples['meta_info']['src_image'] for batched_samples in batched_samples_list]
            batched_src_kpts_list =  [ tf.reshape(batched_samples['meta_info']['src_keypoints'],[-1,17,3])  for batched_samples in batched_samples_list]
            batched_src_bboxes_list = [ batched_samples['meta_info']['src_bbox'] for batched_samples in batched_samples_list]
            batched_src_img_ids_list = [ batched_samples['meta_info']['image_id'] for batched_samples in batched_samples_list]
            batched_src_ids_list = [ batched_samples['meta_info']['id'] for batched_samples in batched_samples_list]

            vis_fn(
                img = batched_images_list,
                y_pred = batched_pred_list,
                decode_pred  = batched_decode_pred_list,
                encode_true  = encoded_trues_list,
                kpts = batched_kpts_list,
                bboxes  = None,
                labels = None,
                src_img  = batched_src_imgs_list,
                src_bboxes  = batched_src_bboxes_list,
                src_kpts  = batched_src_kpts_list,
                src_labels  = None,
                src_img_id  = batched_src_img_ids_list,
                src_ids = batched_src_ids_list,
            )

        # Method 2 : using  parser to get required data
            batched_samples_list = [ model.predict(batched_samples) for batched_samples in batched_samples_list]
            vis_fn(
                batched_samples_list = batched_samples_list,
            )

    """
    def __init__(
            self, 
            simicc_split_xy_dim : Tuple[int] = (192*2, 256*2),
            **kwargs
    ):
        super().__init__(**kwargs)
        self.simicc_split_xy_dim = simicc_split_xy_dim
        
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
        #print(src_ids, src_ids.shape, src_ids.shape.rank)



        img_size_hw = (0,0) if np_resized_img is None else  np_resized_img.shape[:2]#(256,192)
        src_img_size_hw = (0,0)  if np_src_img is None else np_src_img.shape[:2]
        hm_2d_size_hw = (0, 0)
        hm_1d_num = 0
        hm_1d_size = 5
        hm_1d_intervel_row = 5
        hm_1d_intervel_col = 5
        subplot_intervel = 30 
       

        'encode data'
        plot_hm_true = False if encode_true is None else True
        if plot_hm_true :
            simcc_xy_heatmaps_true   = encode_true
            x_heatmaps_true , y_heatmaps_true = tf.split(simcc_xy_heatmaps_true, num_or_size_splits=[*self.simicc_split_xy_dim], axis=-1)
            hm_2d_size_hw = (y_heatmaps_true.shape[1], x_heatmaps_true.shape[1])
            hm_1d_num = x_heatmaps_true.shape[0] 


        'pred data (model output)'
        plot_hm_pred = False if y_pred is None else True
        if plot_hm_pred :
            simcc_xy_heatmaps_pred   = y_pred
            x_heatmaps_pred , y_heatmaps_pred = tf.split(simcc_xy_heatmaps_pred, num_or_size_splits=[*self.simicc_split_xy_dim], axis=-1)
            hm_2d_size_hw = (y_heatmaps_pred.shape[1], x_heatmaps_pred.shape[1])
            hm_1d_num = x_heatmaps_pred.shape[0] 

    

        "one simcc_hm_size_hw including 2d and 1d heatmap"
        simcc_hm_size_hw = (
           hm_2d_size_hw[0]//2 + hm_1d_intervel_row +10+(hm_1d_intervel_row+hm_1d_size)*hm_1d_num,
           (hm_2d_size_hw[1]//2 + hm_1d_intervel_col + (hm_1d_intervel_col+hm_1d_size)*hm_1d_num + subplot_intervel*2)
        )

        'set GridSpec'
        grid_rows = max(simcc_hm_size_hw[0] , img_size_hw[0] + subplot_intervel + src_img_size_hw[0]//2)
        grid_cols = simcc_hm_size_hw[1]*(plot_hm_pred+plot_hm_true) + max( img_size_hw[1]*2 + subplot_intervel, src_img_size_hw[1]//2)
        gs = gridspec.GridSpec(grid_rows, grid_cols)
        'set Image Size'
        fig_size_wh  = (grid_cols//40, grid_rows//40 ) #((hm_2d_size_hw[0]+extra_rows)//40, (hm_2d_size_hw[1]+extra_cols)//40)

        plt.figure(figsize=fig_size_wh)
        if  plot_hm_true:
            self.xy_heatmaps_plot(
                title = "simcc_true ",
                gs = gs,
                x_heatmaps = x_heatmaps_true, 
                y_heatmaps = y_heatmaps_true, 
                lt_start_gxy  = (0,0), 
                hm_1d_size = hm_1d_size,
                hm_1d_intervel_row = hm_1d_intervel_row,
                hm_1d_intervel_col = hm_1d_intervel_col,
            )
        if plot_hm_pred:
            a = simcc_hm_size_hw[1]*plot_hm_true
            self.xy_heatmaps_plot(
                title = "simcc_pred ",
                gs = gs,
                x_heatmaps = x_heatmaps_pred, 
                y_heatmaps = y_heatmaps_pred, 
                lt_start_gxy  = (a,0), 
                hm_1d_size = hm_1d_size,
                hm_1d_intervel_row = hm_1d_intervel_row,
                hm_1d_intervel_col = hm_1d_intervel_col,
            )   

        'resized_image'
        if all(img_size_hw) :
            start_x = simcc_hm_size_hw[1]*(plot_hm_pred+plot_hm_true)
            self.base_plot(
                title = f"Resized Image with gt-kpts",
                image = np_resized_img,
                kpts = kpts,
                bboxes = None,
                subplot_spec = gs[:img_size_hw[0], start_x:start_x+img_size_hw[1]]

            )
        'resized_image # 2'
        #start_x = (hm_2d_size_hw[1]//2+subplot_intervel+hm_1d_num*(hm_1d_intervel_col+hm_1d_size)+subplot_intervel)
        if all(img_size_hw) :
            start_x += img_size_hw[1] + subplot_intervel
            self.base_plot(
                title = f"Resized Image with decoded-kpts",
                image = np_resized_img,
                kpts = decode_pred,
                bboxes = None,
                subplot_spec = gs[:img_size_hw[0], start_x:start_x+img_size_hw[1]]

            )
        'src_image'
        if all(src_img_size_hw) :
            start_x = simcc_hm_size_hw[1]*(plot_hm_pred+plot_hm_true)
            start_y = img_size_hw[0]+subplot_intervel
            extra_text = f" ---(img_id : {image_id}, id : {instance_id} )"
            self.base_plot(
                title = f"Src Image" + extra_text,
                image = np_src_img,
                kpts = src_kpts,
                bboxes = src_bboxes,
                subplot_spec = gs[start_y:start_y + src_img_size_hw[0]//2, start_x:start_x+src_img_size_hw[1]//2]

            )



# class Vis_TopdownPoseSimCCLabelCodec(BasePoseVisFun):
#     VERSION = '1.0.0'

#     R"""  Vis_TopdownPoseSimCCLabelCodec
    
#     i.e. :
#         batch_val_dataset = val_tfds_builder.GenerateTargets(
#             test_mode=True,     
#             unpack_x_y_sample_weight= False, 
#         )

#         batched_samples_list = [ model.encoder(feat) for feat in batch_val_dataset.take(2)]
#         encoded_trues_list = [ batched_samples['y_true'] for batched_samples in batched_samples_list]
#         batched_images_list = [ batched_samples['image'] for batched_samples in batched_samples_list]
#         batched_kpts_list = [ batched_samples['kps'] for batched_samples in batched_samples_list]
#         batched_pred_list = [ model(batched_samples['image']) for batched_samples in batched_samples_list]
#         batched_decode_pred_list = [ model.decoder(batched_samples) for batched_samples in batched_pred_list]
#         batched_src_imgs_list = [ batched_samples['meta_info']['src_image'] for batched_samples in batched_samples_list]
#         batched_src_kpts_list =  [ tf.reshape(batched_samples['meta_info']['src_keypoints'],[-1,17,3])  for batched_samples in batched_samples_list]
#         batched_src_bboxes_list = [ batched_samples['meta_info']['src_bbox'] for batched_samples in batched_samples_list]
#         batched_src_img_ids_list = [ batched_samples['meta_info']['image_id'] for batched_samples in batched_samples_list]
#         batched_src_ids_list = [ batched_samples['meta_info']['id'] for batched_samples in batched_samples_list]



#         vis_fn = Vis_TopdownPoseSimCCLabelCodec(
#             simicc_split_xy_dim  = (192*2, 256*2),
#             figsize = (24, 8),
#             sel_batch_ids  = [0]
#         )
#         vis_fn(
#             img = batched_images_list,
#             y_pred = batched_pred_list,
#             decode_pred  = batched_decode_pred_list,
#             encode_true  = encoded_trues_list,
#             kpts = batched_kpts_list,
#             bboxes  = None,
#             labels = None,
#             src_img  = batched_src_imgs_list,
#             src_bboxes  = batched_src_bboxes_list,
#             src_kpts  = batched_src_kpts_list,
#             src_labels  = None,
#             src_img_id  = batched_src_img_ids_list,
#             src_ids = batched_src_ids_list,
#         )

#     """
#     def __init__(
#             self, 
#             simicc_split_xy_dim : Tuple[int] = (192*2, 256*2),
#             **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.simicc_split_xy_dim = simicc_split_xy_dim


#     def call(
#         self, 
#         img : Tensor,
#         y_pred : Optional[Tensor] = None,
#         decode_pred : Optional[Tensor] = None,
#         encode_true : Optional[Tensor] = None,
#         kpts : Optional[Tensor] = None,
#         bboxes : Optional[Tensor] = None,
#         labels : Optional[Tensor] = None,
#         src_img : Optional[Tensor] = None,
#         src_kpts : Optional[Tensor] = None,
#         src_bboxes : Optional[Tensor] = None,
#         src_labels : Optional[Tensor] = None,
#         src_img_id : Optional[Tensor] = None,
#         src_ids : Optional[Tensor] = None,
#         **kwargs
#     )->None:
        
#         r"""

#         """
#         '''

#         '''
#         np_resized_img =  None if img  is None else img.numpy()
#         np_src_img = None if src_img is None else src_img.numpy()  
#         image_id = '' if src_img_id is None else src_img_id
#         instance_id = '' if src_ids is None else src_ids



#         img_size_hw = (0,0) if np_resized_img is None else  np_resized_img.shape[:2]#(256,192)
#         src_img_size_hw = (0,0)  if np_src_img is None else np_src_img.shape[:2]
#         hm_2d_size_hw = (0, 0)
#         hm_1d_num = 0
#         hm_1d_size = 5
#         hm_1d_intervel_row = 5
#         hm_1d_intervel_col = 5
#         subplot_intervel = 30 
       

#         'encode data'
#         plot_hm_true = False if encode_true is None else True
#         if plot_hm_true :
#             simcc_xy_heatmaps_true   = encode_true
#             x_heatmaps_true , y_heatmaps_true = tf.split(simcc_xy_heatmaps_true, num_or_size_splits=[*self.simicc_split_xy_dim], axis=-1)
#             hm_2d_size_hw = (y_heatmaps_true.shape[1], x_heatmaps_true.shape[1])
#             hm_1d_num = x_heatmaps_true.shape[0] 


#         'pred data (model output)'
#         plot_hm_pred = False if y_pred is None else True
#         if plot_hm_pred :
#             simcc_xy_heatmaps_pred   = y_pred
#             x_heatmaps_pred , y_heatmaps_pred = tf.split(simcc_xy_heatmaps_pred, num_or_size_splits=[*self.simicc_split_xy_dim], axis=-1)
#             hm_2d_size_hw = (y_heatmaps_pred.shape[1], x_heatmaps_pred.shape[1])
#             hm_1d_num = x_heatmaps_pred.shape[0] 

    

#         "one simcc_hm_size_hw including 2d and 1d heatmap"
#         simcc_hm_size_hw = (
#            hm_2d_size_hw[0]//2 + hm_1d_intervel_row +10+(hm_1d_intervel_row+hm_1d_size)*hm_1d_num,
#            (hm_2d_size_hw[1]//2 + hm_1d_intervel_col + (hm_1d_intervel_col+hm_1d_size)*hm_1d_num + subplot_intervel*2)
#         )

#         'set GridSpec'
#         grid_rows = max(simcc_hm_size_hw[0] , img_size_hw[0] + subplot_intervel + src_img_size_hw[0]//2)
#         grid_cols = simcc_hm_size_hw[1]*(plot_hm_pred+plot_hm_true) + max( img_size_hw[1]*2 + subplot_intervel, src_img_size_hw[1]//2)
#         gs = gridspec.GridSpec(grid_rows, grid_cols)
#         'set Image Size'
#         fig_size_wh  = (grid_cols//40, grid_rows//40 ) #((hm_2d_size_hw[0]+extra_rows)//40, (hm_2d_size_hw[1]+extra_cols)//40)

#         plt.figure(figsize=fig_size_wh)
#         if  plot_hm_true:
#             self.xy_heatmaps_plot(
#                 title = "simcc_true ",
#                 gs = gs,
#                 x_heatmaps = x_heatmaps_true, 
#                 y_heatmaps = y_heatmaps_true, 
#                 lt_start_gxy  = (0,0), 
#                 hm_1d_size = hm_1d_size,
#                 hm_1d_intervel_row = hm_1d_intervel_row,
#                 hm_1d_intervel_col = hm_1d_intervel_col,
#             )
#         if plot_hm_pred:
#             a = simcc_hm_size_hw[1]*plot_hm_true
#             self.xy_heatmaps_plot(
#                 title = "simcc_pred ",
#                 gs = gs,
#                 x_heatmaps = x_heatmaps_pred, 
#                 y_heatmaps = y_heatmaps_pred, 
#                 lt_start_gxy  = (a,0), 
#                 hm_1d_size = hm_1d_size,
#                 hm_1d_intervel_row = hm_1d_intervel_row,
#                 hm_1d_intervel_col = hm_1d_intervel_col,
#             )   

#         'resized_image'
#         if all(img_size_hw) :
#             start_x = simcc_hm_size_hw[1]*(plot_hm_pred+plot_hm_true)
#             self.base_plot(
#                 title = f"Resized Image with gt-kpts",
#                 image = np_resized_img,
#                 kpts = kpts,
#                 bboxes = None,
#                 subplot_spec = gs[:img_size_hw[0], start_x:start_x+img_size_hw[1]]

#             )
#         'resized_image # 2'
#         #start_x = (hm_2d_size_hw[1]//2+subplot_intervel+hm_1d_num*(hm_1d_intervel_col+hm_1d_size)+subplot_intervel)
#         if all(img_size_hw) :
#             start_x += img_size_hw[1] + subplot_intervel
#             self.base_plot(
#                 title = f"Resized Image with decoded-kpts",
#                 image = np_resized_img,
#                 kpts = decode_pred,
#                 bboxes = None,
#                 subplot_spec = gs[:img_size_hw[0], start_x:start_x+img_size_hw[1]]

#             )
#         'src_image'
#         if all(src_img_size_hw) :
#             start_x = simcc_hm_size_hw[1]*(plot_hm_pred+plot_hm_true)
#             start_y = img_size_hw[0]+subplot_intervel
#             extra_text = f" ---(img_id : {image_id}, id : {instance_id} )"
#             self.base_plot(
#                 title = f"Src Image" + extra_text,
#                 image = np_src_img,
#                 kpts = src_kpts,
#                 bboxes = src_bboxes,
#                 subplot_spec = gs[start_y:start_y + src_img_size_hw[0]//2, start_x:start_x+src_img_size_hw[1]//2]

#             )

#---------------------------------------------------------------------

#----------------------------------------------------------------------
            
# def Vis_TopdownPoseSimCCLabelCodec(
#     batched_samples : Union[Dict, List],   
#     batch_ids : int =0,
#     batched_decoded_kpts : Optional[Union[tf.Tensor, List]] = None,
#     show_kpts : bool = True
# ):
#     import matplotlib.pyplot as plt
#     import matplotlib.gridspec as gridspec
    
#     def plot(
#         title,  image, kpts=None, bbox=None, subplot_spec=(1,1,1),**kwargs
#     ):  
#         if type(subplot_spec)==tuple:
#             plt.subplot(*subplot_spec,**kwargs)
#         else:
#             plt.subplot(subplot_spec, **kwargs)

#         plt.title(
#             title, fontsize = 12
#         )
#         plt.imshow(image)
#         if kpts is not None :
#             for i in range(0,kpts.shape[0]):
#                 kpt_x = int((kpts[i,0]))
#                 kpt_y = int((kpts[i,1]))
#                 plt.scatter(kpt_x,kpt_y)
    
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
    
#     for ith, features in enumerate(batched_samples):
#         for batch_id in batch_ids :
#             resized_image = features["image"][batch_id].numpy()
#             kpts = features['kps'][batch_id]
#             if show_kpts :
#                 print(kpts)
#             if batched_decoded_kpts is not None:
#                decoded_kpts = batched_decoded_kpts[ith][batch_id] 
        
#             if  features.get('meta_info',None) is not None :
#                 meta_info = features['meta_info']
#                 image_id = meta_info['image_id'][batch_id]
#                 instance_id = meta_info['id'][batch_id]
#                 src_image = meta_info['src_image'][batch_id].numpy()
#                 src_bbox = meta_info['src_bbox'][batch_id] 
#                 src_kpts = meta_info['src_keypoints'][batch_id]
#                 src_kpts = tf.reshape(src_kpts,[*(kpts.shape)])


#             simcc_xy_heatmaps = features['y_true'][batch_id] #(17, simcc_x+simcc_y)
#             x_heatmaps , y_heatmaps = tf.split(simcc_xy_heatmaps, num_or_size_splits=[192*2,256*2], axis=-1)
#             heatmaps = x_heatmaps[:,None,:]*y_heatmaps[:,:,None] #(17,simcc_y, simcc_x)
#             heatmap2d = tf.reduce_max(heatmaps, axis=0).numpy() #(simcc_y, simcc_x)


#             ''
#             hm_2d_size_hw = heatmaps.shape[1:] #(512,384)
#             img_size_hw = resized_image.shape[:2] #(256,192)
#             src_img_size_hw = src_image.shape[:2]
#             hm_1d_num = kpts.shape[0]
#             hm_1d_size = 5
#             hm_1d_intervel_row = 5
#             hm_1d_intervel_col = 5
#             subplot_intervel = 30 

#             'set GridSpec'
#             grid_rows = hm_2d_size_hw[0]//2 + max(hm_1d_intervel_row+10+(hm_1d_intervel_row+hm_1d_size)*hm_1d_num, img_size_hw[0]+subplot_intervel)
#             grid_cols = hm_2d_size_hw[1]//2 + hm_1d_intervel_row+ (hm_1d_intervel_row+hm_1d_size)*hm_1d_num + subplot_intervel + img_size_hw[1] +src_img_size_hw[1]//2
#             gs1 = gridspec.GridSpec(grid_rows, grid_cols)
#             'set Image Size'
#             fig_size_wh  = (grid_cols//40, grid_rows//40 ) #((hm_2d_size_hw[0]+extra_rows)//40, (hm_2d_size_hw[1]+extra_cols)//40)
#             plt.figure(figsize=fig_size_wh)
#             'Heatmap2D'
#             plot(
#                 f"Heatmap2D ",
#                 heatmap2d,
#                 kpts = None,
#                 bbox = None,
#                 subplot_spec = gs1[:hm_2d_size_hw[0]//2, :hm_2d_size_hw[1]//2]
#             )
#             'resized_image'
#             start_x = hm_2d_size_hw[1]//2+subplot_intervel+hm_1d_num*(hm_1d_intervel_col+hm_1d_size)+subplot_intervel
#             plot(
#                 f"Resized Image",
#                 resized_image,
#                 kpts = kpts,
#                 bbox = None,
#                 subplot_spec = gs1[:img_size_hw[0], start_x:start_x+img_size_hw[1]]

#             )
#             'source image  (option)'
#             if 'meta_info' in locals():
#                 extra_text = f" ---(img_id : {image_id}, id : {instance_id} )"
#                 start_y = img_size_hw[0]+subplot_intervel #256 + 20
#                 plot(
#                     f"src Image  " + extra_text,
#                     src_image,
#                     kpts = src_kpts,
#                     bbox = src_bbox,
#                     subplot_spec = gs1[start_y:start_y+src_img_size_hw[0]//2, start_x:start_x+src_img_size_hw[1]//2],
#                     aspect='equal'
#                 )
#             'decode image (option)'
#             if 'decoded_kpts' in locals():
#                 start_x = start_x+img_size_hw[1]+subplot_intervel
#                 # plt.subplot(gs1[start_y:start_y+img_size_hw[0], start_x:start_x+img_size_hw[1]])
#                 # plt.title('decode_image')
#                 # plt.imshow(resized_image, aspect='auto')
#                 plot(
#                     f"Decoded Results",
#                     resized_image,
#                     kpts = decoded_kpts,
#                     bbox = None,
#                     subplot_spec = gs1[:img_size_hw[0] , start_x: start_x+img_size_hw[1]]
#                 ) 
#             'Heatmap1D_xy'
#             start_y = hm_2d_size_hw[0]//2+hm_1d_intervel_row+10
#             start_x = hm_2d_size_hw[1]//2+hm_1d_intervel_col
#             for i in range(heatmaps.shape[0]) :
#                 #x_heatmap = tf.reduce_max(heatmaps[i],axis=0,keepdims=True)
#                 x_heatmap = x_heatmaps[i][None,:]
#                 ax2 = plt.subplot(gs1[start_y:start_y+hm_1d_size, :hm_2d_size_hw[1]//2])
#                 plt.imshow(x_heatmap, aspect='auto')
#                 ax2.set_yticks([])
#                 if i!=heatmaps.shape[0]-1 :
#                     ax2.set_xticks([])
#                 ax2.set_ylabel(f'{i}',fontweight ='bold', fontsize=7).set_color('red')
#                 start_y += hm_1d_intervel_row+hm_1d_size

               
#                 #y_heatmap = tf.reduce_max(heatmaps[i],axis=1,keepdims=True)
#                 y_heatmap = y_heatmaps[i][:, None] 
#                 ax2 = plt.subplot(gs1[:hm_2d_size_hw[0]//2, start_x:start_x+hm_1d_size])
#                 plt.imshow(y_heatmap, aspect='auto')
#                 if i!=heatmaps.shape[0]-1 :
#                     ax2.set_yticks([])
#                 else:
#                     ax2.yaxis.set_label_position("right")
#                     ax2.yaxis.tick_right()
#                 ax2.set_xticks([])
#                 ax2.set_xlabel(f'{i}',fontweight ='bold', fontsize=7).set_color('red') 
#                 start_x += hm_1d_intervel_col+hm_1d_size
    
# @CODECS.register_module()
# class  UDP_SimCCLabel(BaseKeypointCodec):
#     def __init__(self, 
#             sigma_xy =(4.9,5.66), 
#             image_size_xy=(192,256), 
#             simcc_split_ratio=2.,
#             normalize = False,
#             use_udp = True):
        
#         """ Tags the result of function by setting _is_zeros_tensor attribute.
#         Input kps : (x, y, vis)
#         Output heatmap_size : (w,h) or (x,y)


#         Simple Coordinate Classification (simcc)
#         """
        
#         self.image_size_xy = image_size_xy
#         self.simcc_ratio = simcc_split_ratio
#         self.target_size_xy = (int(image_size_xy[0]*self.simcc_ratio),int(image_size_xy[1]*self.simcc_ratio) )
#         self.normalize = normalize 
    
#         self.sigma_xy = tf.constant(sigma_xy, dtype=self.compute_dtype) #(2,)
#         self.sigma_square_xy = tf.math.square(self.sigma_xy)  #(2,)
#         self.temp_size_xy = self.sigma_xy*3. #(2,)

#         if self.normalize :
#             self.sigma_sqrt_2pi = self.sigma_xy*tf.math.sqrt(2.*math.pi) #(2,)

#         if use_udp :
#             self.simcc_split_ratio =(tf.cast(self.target_size_xy, dtype=self.compute_dtype) -1.)/(tf.cast(self.image_size_xy,dtype=self.compute_dtype)-1.)
#         else:
#             self.simcc_split_ratio =(tf.cast(self.target_size_xy, dtype=self.compute_dtype))/(tf.cast(self.image_size_xy,dtype=self.compute_dtype))

#         print(f"target_size_xy : {self.target_size_xy}")
#         print(f"sigma_xy : {self.sigma_xy}")
#         print(f"simdr_split_ratio : {self.simcc_split_ratio}")

#         def target_xy_gen():
#             coords_x = tf.cast(tf.range(0, self.target_size_xy[0]), dtype=self.compute_dtype)   #(coords_x,)
#             coords_y = tf.cast(tf.range(0, self.target_size_xy[1]), dtype=self.compute_dtype)   #(coords_y,)
#             return coords_x, coords_y 

#         self.simcc_coords_x, self.simcc_coords_y = target_xy_gen()



#         'kps confidence threhold config'
#         #self.val_thr = tf.ones(shape=(17,1), dtype=self.compute_dtype)*0.0

#     def batch_encode(self,
#                     data : Dict[str,tf.Tensor],
#                     use_vectorized_map :bool=False) -> Dict[str,tf.Tensor]:
#         'add new keys for encode results'
#         #data['y_true'] = tf.zeros(shape=(,self.target_size_xy[1]+self.target_size_xy[0]), dtype=self.compute_dtype)
#         #data['sample_weight'] = tf.zeros(shape=(self.num_kps,3), dtype=self.compute_dtype)
#         kps_true = data['kps']

#         'kps transform to simcc from image size(256x192)'
#         mu_xy = kps_true[:,:,:2]*self.simcc_split_ratio #(b,17,2)

#         lu = tf.cast( kps_true[:,:,:2] - self.temp_size_xy, dtype=tf.int32)   #(b,17,2,)
#         rb = tf.cast( kps_true[:,:,:2] + self.temp_size_xy+1, dtype=tf.int32)  #(b,17,2,)

#         mask_lu = tf.greater_equal(lu, self.image_size_xy) #(b, 17,2,)
#         mask_rb = tf.less(rb, [0,0]) #(b, 17,2,)
#         mask_vis = tf.expand_dims(tf.equal( kps_true[:, :, 2], 0.), axis=-1) #(b,17,1)
#         mask = tf.reduce_any(tf.concat([mask_lu, mask_rb, mask_vis],axis=-1), axis=-1, keepdims=True) #(b,17,5) => (b,17,1)

#         target_weights = tf.where(mask,0.,1.) #(17,1)

#         target_x = tf.math.squared_difference( self.simcc_coords_x[None,None,:], mu_xy[:,:,0,None]) # (b, 17, None)x(None, None, coords_x)=>(b, 17, coords_x,)
#         target_y = tf.math.squared_difference( self.simcc_coords_y[None,None,:], mu_xy[:,:,1,None]) #(b, 17, coords_y,)
#         target_x = tf.math.exp( -target_x/(2.*self.sigma_square_xy[0]))#(b,17, coords_x,)
#         target_y = tf.math.exp( -target_y/(2.*self.sigma_square_xy[1])) #(b, 17, coords_y,)
#         if self.normalize :
#             target_x /= self.sigma_sqrt_2pi[0]
#             target_y /= self.sigma_sqrt_2pi[1]
    
#         target_x = target_x*target_weights #(b,17, coords_x)*(b,17, 1)=>(b,17, coords_x)
#         target_y = target_y*target_weights
#         target_xy = tf.concat([target_x,target_y], axis=-1)  #(b, 17, coords_x+coords_y)
#         sample_weights = tf.concat([kps_true[:, :, :2], target_weights], axis=-1)    

#         data['y_true'] =  target_xy  #simicc--encode : (b, 17, coords_x+coords_y)
#         data['sample_weight'] = sample_weights #coordinate based on model's input dimemsion
#         return data

#     def encode(self, kps_true : Tensor)->Tuple[Tensor, Tensor]: 
#         """

#         kps_true : (17,3)
#         """ 
#         'kps transform to simcc from image size(256x192)'
#         mu_xy = kps_true[:,:2]*self.simcc_split_ratio #(17,2)

#         lu = tf.cast( kps_true[:,:2] - self.temp_size_xy, dtype=tf.int32)   #(17,2,)
#         rb = tf.cast( kps_true[:,:2] + self.temp_size_xy+1, dtype=tf.int32)  #(17,2,)

#         mask_lu = tf.greater_equal(lu, self.image_size_xy) #(17,2,)
#         mask_rb = tf.less(rb, [0,0]) #(17,2,)
#         mask_vis = tf.expand_dims(tf.equal( kps_true[:, 2], 0.), axis=-1) #(17,1)
#         mask = tf.reduce_any(tf.concat([mask_lu, mask_rb, mask_vis],axis=-1), axis=-1, keepdims=True) #(17,1)

#         target_weights = tf.where(mask,0.,1.) #(17,1)

#         target_x = tf.math.squared_difference( self.simcc_coords_x[None,:], mu_xy[:,0,None]) #(17, coords_x,)
#         target_y = tf.math.squared_difference( self.simcc_coords_y[None,:], mu_xy[:,1,None]) #(17, coords_y,)
#         target_x = tf.math.exp( -target_x/(2.*self.sigma_square_xy[0]))#(17, coords_x,)
#         target_y = tf.math.exp( -target_y/(2.*self.sigma_square_xy[1])) #(17, coords_y,)
#         if self.normalize :
#             target_x /= self.sigma_sqrt_2pi[0]
#             target_y /= self.sigma_sqrt_2pi[1]
    
#         target_x = target_x*target_weights
#         target_y = target_y*target_weights
#         target_xy = tf.concat([target_x,target_y], axis=-1)

#         sample_weights = tf.concat([kps_true[:, :2], target_weights], axis=-1)
#         return target_xy, sample_weights
    
#     def batch_simcc2hm(self, batch_encoded : Tensor, pool_size : Optional[Union[int,None]]=None) -> Tensor:

#         assert batch_encoded.ndim ==3,'batch_encoded.ndim must be 3@batch_heatmap_2d'
#         simcc_x, simcc_y = tf.split(batch_encoded, (self.target_size_xy[0], self.target_size_xy[1]), axis=-1)
#         if pool_size is None :
#             hm_yx_NCHW = simcc_x[:,:,None,:]*simcc_y[:,:,:,None]
#             hm_yx_NHWC = tf.transpose(hm_yx_NCHW, perm=[0,2,3,1])
#         else:
#             simcc_x = tf.transpose(simcc_x,perm=[0,2,1])
#             simcc_y = tf.transpose(simcc_y,perm=[0,2,1])
#             simcc_x = tf.nn.avg_pool1d(simcc_x,ksize=pool_size,strides=pool_size, padding='SAME')
#             simcc_y = tf.nn.avg_pool1d(simcc_y,ksize=pool_size,strides=pool_size, padding='SAME')
#             hm_yx_NHWC = simcc_x[:,None,:,:]*simcc_y[:,:,None,:]
#         return hm_yx_NHWC


#     def decode(self, encoded : Tensor, 
#                val_thr : Optional[Tensor]=None) -> Tensor:

#         return self.batch_decode(encoded)
    

#     def batch_decode(self, 
#                     batch_encoded : Tensor, 
#                     val_thr : Optional[Tensor]=None) ->Tensor:
        
#         if val_thr is None :
#             val_thr = tf.ones(shape=(17,1), dtype=self.compute_dtype)*0.0

#         'to do assert function'
#         'kps_from_simcc'

#         'kps confidence threhold config'
#         #val_thr = tf.ones(shape=(17,1), dtype=self.compute_dtype)*0.0

#         batch_simcc_x, batch_simcc_y = tf.split(batch_encoded, num_or_size_splits=[*self.target_size_xy], axis=-1) #(b,17,192*2)

#         argmax_x = tf.argmax(batch_simcc_x, axis=-1) #(b,17)
#         argmax_y = tf.argmax(batch_simcc_y, axis=-1) #(b,17)

#         argmax_xy = tf.cast(tf.stack([argmax_x,argmax_y], axis=-1), dtype=self.compute_dtype)/self.simcc_split_ratio  #(b,17,2)
        
#         max_val_x  = tf.math.reduce_max(batch_simcc_x, axis=-1)  #(b,17)
#         max_val_y  = tf.math.reduce_max(batch_simcc_y, axis=-1)  #(b,17)
#         val = tf.expand_dims( tf.math.minimum(max_val_x,max_val_y), axis=-1) #(b,17,1)

#         mask =  tf.greater(val, val_thr) #(b,17,1)
#         batch_keypoints = tf.where(mask, argmax_xy, 0.) #(1,17,2)
#         batch_keypoints = tf.concat([batch_keypoints, val], axis=-1) #(1,17,3)    

#         return batch_keypoints