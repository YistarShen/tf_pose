from typing import Dict, List, Optional, Tuple, Union, Sequence
import matplotlib.pyplot as plt
from tensorflow import Tensor
import tensorflow as tf
from .base_codec import BaseCodec
from lib.Registers import CODECS
from lib.datasets.transforms.utils import PackInputTensorTypeSpec
from lib.visualization.base_vis import BasePoseVisFun
#---------------------------------------------------------------------
#
#----------------------------------------------------------------------
def transform_preds(kps,
                    aspect_scale_xy,
                    offset_padding_xy, 
                    bbox_lt_xy):
    
    coords_xy = kps[...,:2]
    score =  kps[...,2:]
    mask = tf.cast(tf.math.count_nonzero(coords_xy, axis=-1), dtype=tf.bool)
    kps_xy = (coords_xy-offset_padding_xy[:,None,:])*aspect_scale_xy[:,None,:] + bbox_lt_xy[:,None,:] #(1, 17,2) 
    kps_xy = tf.where(mask, kps_xy, 0.)
    return tf.concat([kps_xy, score], axis=-1)


#---------------------------------------------------------------------
#
#----------------------------------------------------------------------
@CODECS.register_module()
class MSRAHeatmapCodec(BaseCodec):
    VERSION = '2.2.0'
    ENCODER_USE_PRED = False
    ENCODER_GEN_SAMPLE_WEIGHT = True
    r""" MSRAHeatmap used in heatmap base 2D pose detection,i.e. ViTPose/HRNet
    date : 2024/2/19
    author : Dr. Shen 

        Input kps : (x, y, vis)
        Output heatmap_size : (w,h) or (x,y)
    
        use_udp (bool): 
        num_kps (int): 
        sigma (int) :
        target_size_xy (tuple) :
        heatmap_size_xy (tuple) :
        hm_thr (List[float] | tf.tensor):  threshold value of heatmap to get effective kps

    batch_decode :
         

    Example :
        coco_17kps_256x192 :
        - codec_cfg = dict(
                        use_udp = True, 
                        num_kps = 17,
                        sigma = 2,
                        target_size_xy = (192,256),
                        heatmap_size_xy = (48, 64),
                        hm_thr= [0.75, 0.75, 0.75, 0.75, 0.75,
                                0.25, 0.25, 0.25, 0.25, 0.25,
                                0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
                    )
        coco_17kps_384x288 :
        - codec_cfg = dict(
                        use_udp = True, 
                        num_kps = 17,
                        sigma = 3,
                        target_size_xy = (288,384),
                        heatmap_size_xy = (72, 96),
                        hm_thr= tf.ones(shape=(17,), dtype=self.compute_dtype)*0.5
                    )
    dict(type= '',
        num_kps = 17,
        target_size_xy = (288,384)
        use_udp = True
        encode_cfg = dict(sigma = 3. 
                          heatmap_size_xy =  (72, 96),
                          normalize = False
        
        ),
        decode_cfg = dict(kernel = 3,
                          hm_thr = []
                          src_img_frame = False,
                          use_vectorized_map = False
        
        ),
    )
    """   

    def __init__(self, 
            use_udp : bool = True,
            num_kps : int =17, 
            sigma =3, 
            target_size_xy : Tuple[int, int] = (288,384), 
            heatmap_size_xy : Tuple[int, int] = (72, 96),
            hm_thr : Union[tf.Tensor, List[float]] = None,
            nms_kernel : int = 5,
            **kwargs):
        super().__init__(**kwargs)

        'set cfg for tf.map_fn'

        self.use_udp = use_udp
        self.num_kps = num_kps
        self.heatmap_size_xy = heatmap_size_xy
        self.target_size_xy = target_size_xy
        self.sigma = sigma
        self.radius = self.sigma * 3
        self.sigma_square = tf.math.square(
            tf.cast(self.sigma,dtype=self.compute_dtype)
        )

        'enable UDP or no disable'
        if use_udp :
            self.feat_stride = ( 
                tf.cast(target_size_xy,dtype=self.compute_dtype)-1. ) / ( tf.cast(heatmap_size_xy,dtype=self.compute_dtype)-1.
            )
        else:
            self.feat_stride = ( 
                tf.cast(target_size_xy,dtype=self.compute_dtype)) / ( tf.cast(heatmap_size_xy,dtype=self.compute_dtype)
            )
        
        'set hm_thr to decode'
        if isinstance(hm_thr, (Sequence, tf.Tensor)):
            self.hm_thr = tf.cast(hm_thr, dtype=self.compute_dtype)
            assert  self.hm_thr.shape[0] == num_kps, \
             f"hm_thr.shape[0] must be equal to {num_kps}, but got {self.hm_thr.shape[0]} @MSRAHeatmap"  
        else:
            self.hm_thr = tf.zeros(shape=(num_kps,), dtype=self.compute_dtype) 
        'set nms_kernel to decode'
        self.nms_kernel = nms_kernel
                  
    
        'basic grid_map and  core_xy'

        def grid_map_gen(heat_range_xy : tuple):
            x_core = heat_range_xy[0] // 2
            y_core = heat_range_xy[1] // 2
            grid_coords_y = tf.cast(tf.range(0, heat_range_xy[1]), dtype=self.compute_dtype)   #(grid_coords_y,)
            grid_coords_x = tf.cast(tf.range(0, heat_range_xy[0]), dtype=self.compute_dtype)   #(grid_coords_x,)
            grid_x, grid_y = tf.meshgrid(grid_coords_x, grid_coords_y)  # (grid_coords_x,grid_height)
            grid_x = tf.expand_dims(grid_x, axis=-1)   # (grid_coords_x,grid_height,1)
            grid_y = tf.expand_dims(grid_y, axis=-1)
            grid_xy = tf.concat([grid_x, grid_y], axis=-1)  # (13,13,2)
            return grid_xy, tf.constant([x_core,y_core], dtype=self.compute_dtype)
        
        if self.use_vectorized_map:
            self.grid_map, self.core_xy = grid_map_gen(heatmap_size_xy)
        else:
            'set cfg for tf.map_fn'
            heat_range = self.radius*2+1
            self.grid_map, self.core_xy = grid_map_gen((heat_range, heat_range))
            self.HeatmapTensorSpec =  tf.type_spec_from_value(
                tf.ones(shape=(heatmap_size_xy[1],heatmap_size_xy[0]), dtype=self.compute_dtype)
            )
            self.VisTensorSpec = tf.type_spec_from_value(
                tf.constant(1, dtype=self.compute_dtype)
            ) 
        print(self.grid_map.shape, self.core_xy.shape)


    def _gen_heatmaps(self, flatten_kps_dist, lu, rb, mask):
        ''' 
        hm_kps : (3,)
        hm_kps_int : (3,)
        '''
        def false_fn(): 
            heatmap = tf.zeros(
                shape=(self.heatmap_size_xy[1], self.heatmap_size_xy[0]),dtype=self.compute_dtype
            ) 
            target_weight = tf.constant(0, dtype=self.compute_dtype)
            return heatmap, target_weight
        
        def true_fn():
            #core_xy = self.core_xy + (hm_kps[:2] - tf.cast(hm_kps_int[:2], dtype=self.compute_dtype)) #(2,)
            core_xy = self.core_xy + flatten_kps_dist

            diff_square = tf.math.squared_difference( 
                self.grid_map, core_xy, name=None
            )#(13,13,2)
            diff_square_sum = tf.reduce_sum(
                diff_square,axis=-1
            )#(13,13)
            gaussian_map = tf.math.exp( 
                -diff_square_sum/(2*self.sigma_square) 
            )#(13,13)

            x_gauss_left = tf.math.maximum(0, -lu[0])
            x_gauss_right = tf.math.minimum(rb[0], self.heatmap_size_xy[0]) - lu[0]
            y_gauss_up = tf.math.maximum(0, -lu[1])
            y_gauss_bottom = tf.math.minimum(rb[1], self.heatmap_size_xy[1]) - lu[1]  

            #eff_gaussian_map = gaussian_map[y_gauss_up:y_gauss_bottom, x_gauss_left:x_gauss_right]
            eff_gaussian_map = tf.slice( 
                gaussian_map, 
                begin= [y_gauss_up,x_gauss_left], 
                size = [y_gauss_bottom-y_gauss_up, x_gauss_right-x_gauss_left]
            )
            paddings = tf.stack(
                [ tf.reverse(lu, axis=[0]), tf.reverse(self.heatmap_size_xy-rb, axis=[0])],  axis=-1
            )
            paddings = tf.math.maximum(
                [[0, 0], [0, 0]], paddings 
            )
            heatmap = tf.pad(
                eff_gaussian_map, paddings
            )
      
            target_weight = tf.constant(1, dtype=self.compute_dtype)
            return heatmap, target_weight
        
        heatmap, target_weight = tf.cond(
            mask,
            true_fn,
            false_fn
        )
        return heatmap, target_weight
    
    def _vectorized_gen_heatmaps(self,flatten_hm_kps, flatten_hm_kps_int, mask):
        diff_square = tf.math.squared_difference(
              self.grid_map[None,...], flatten_hm_kps[:,None,None]
        )#(b*17,64,48,2)
        grid_mask = tf.less_equal(
            tf.math.abs(
                flatten_hm_kps_int[:,None,None,:]-tf.cast(self.grid_map[None,...], dtype=tf.int32)
            ), 
            self.radius
        )#(b*17,64,48,2)
        grid_mask = tf.reduce_all(grid_mask, axis=-1) #(b*17, 64,48,)

        diff_square_sum = tf.reduce_sum(
              diff_square,axis=-1
        )#(b*17,64,48)
        heatmap = tf.math.exp(
             -diff_square_sum/(2*self.sigma_square)
        )#(13,13)
        heatmap = tf.where(grid_mask, heatmap, 0.)
        heatmap = tf.where(mask[:,None,None], heatmap, 0.)
        target_weight = tf.cast(mask, dtype=self.compute_dtype)
        return heatmap, target_weight
    

    
    def _gen_targets(self, kps_true):
        # kps_true: (b,17,3)
        vis = tf.reshape(
            kps_true[..., 2:3],(-1,1)
        ) #(b*17,1)
        hm_kps = kps_true[...,:2]/self.feat_stride #(b, 17,2)
        hm_kps_int = tf.cast( hm_kps + 0.5, dtype=tf.int32) # to check (b, 17,2)

        flatten_hm_kps =  tf.reshape(hm_kps,(-1, 2))
        flatten_hm_kps_int =  tf.reshape(hm_kps_int,(-1, 2))

        lu = tf.cast( 
            flatten_hm_kps_int[...,:2] - self.radius, dtype=tf.int32
        ) #(b*17,2)
        rb = tf.cast( 
            flatten_hm_kps_int[...,:2] + self.radius + 1, dtype=tf.int32
        )#(b*17,2)

        #flatten_kps_dist = (flatten_hm_kps[...,:2] - tf.cast(flatten_hm_kps_int[...,:2], dtype=self.compute_dtype))

        cond_lu = tf.greater_equal(lu, self.heatmap_size_xy) #(b*17,2)
        cond_rb = tf.less( rb, 0) #(b*17,2)
        cond_vis = tf.equal(vis, 0.) #(b*17,1)
        cond = tf.concat([cond_lu,cond_rb, cond_vis], axis=-1) #(b*17,5)
        cond = tf.math.reduce_any(cond, axis=-1) #(b*17)
        mask = tf.math.logical_not(cond)  #(b*17)


        'gen heatmap for each keypoints'
        if self.use_vectorized_map :
            flatten_res = self._vectorized_gen_heatmaps(
                flatten_hm_kps, flatten_hm_kps_int, mask
            )#(b*17, 64, 48)

        else:
            flatten_kps_dist = (
                flatten_hm_kps[...,:2] - tf.cast(flatten_hm_kps_int[...,:2], dtype=self.compute_dtype)
            )
            flatten_res = tf.map_fn(
                lambda x : self._gen_heatmaps(x[0], x[1], x[2], x[3]),
                (flatten_kps_dist, lu, rb, mask) ,
                parallel_iterations = self.parallel_iterations,
                fn_output_signature = (
                    self.HeatmapTensorSpec ,
                    self.VisTensorSpec,
                )
            )#(b*17, 64, 48)

        flatten_heatmaps , flatten_target_weight = flatten_res    
        heatmaps = tf.reshape(
            flatten_heatmaps,
            [-1,self.num_kps,self.heatmap_size_xy[1],self.heatmap_size_xy[0]]
        )
        heatmaps = tf.transpose(heatmaps,perm=[0,2,3,1])

        target_weights = tf.reshape(
            flatten_target_weight,[-1,self.num_kps, 1]
        )#(b,17,1)
        # kps_true = tf.concat(
        #     [kps_true[..., :2], target_weights], axis=-1
        # )#(b,17,3)  

        kps_true = tf.concat(
            [kps_true[..., :2]*target_weights, target_weights], axis=-1
        )#(b,17,3)  
       

        if  self.embedded_codec :
            kps_true = tf.stop_gradient(kps_true)
            heatmaps = tf.stop_gradient(heatmaps)

        return (heatmaps,kps_true)

    
    def batch_encode(
        self,
        data : Dict[str,tf.Tensor],
        y_pred  = None
    )-> Dict[str,tf.Tensor]:
        
        heatmaps, kps_true = self._gen_targets(
            tf.cast(data['kps'], dtype=self.compute_dtype)
        )
        data['kps'] = kps_true
        data['y_true'] =  heatmaps #(b,64,48,17)
        data['sample_weight'] = kps_true[...,2]#(b,17)
        return data

  
    def transform_preds(
            self,
            coords_xy, 
            mask,
            data_to_src_img_frame = dict(
                aspect_scale_xy = [1.,1.],
                offset_padding_xy =[0.,0.],
                bbox_lt_xy= [0.,0.]
            ) 
        ):

        data_to_src_img_frame['aspect_scale_xy'] = tf.cast()

        # dict(
        #     aspect_scale_xy = tf.constant([1.,1.], dtype=self.compute_dtype),
        #     offset_padding_xy = tf.constant([0.,0.], dtype=self.compute_dtype),
        #     bbox_lt_xy= tf.constant([0.,0.], dtype=self.compute_dtype)
        # ) 

        
        # assert aspect_scale_xy.shape[1] == 2, f"{aspect_scale_xy.shape[1]}"
        # assert offset_padding_xy.shape[1] == 2, f"{offset_padding_xy.shape[1]}"
        # assert bbox_lt_xy.shape[1] == 2, f"{bbox_lt_xy.shape[1]}"
        # assert coords_xy.shape[1] == mask.shape[1], f"{coords_xy.shape[1]} and {mask.shape[1]}"
        # assert coords_xy.shape.rank == 3, f"{coords_xy.shape.rank }"
        # assert mask.shape.rank == 2, f"{mask.shape.rank}"

        if not isinstance(data_to_src_img_frame, dict):
            raise TypeError(
                "META DATA 'data_to_src_img_frame' must be dict type"
            )
        required_keys = ['aspect_scale_xy', 'offset_padding_xy', 'bbox_lt_xy']
        if all([ x in data_to_src_img_frame.keys() for x in required_keys]) :

            'get required data to do transform back to src img frame'
            required_data= [tf.cast(data_to_src_img_frame[key], dtype=self.compute_dtype) for key in required_keys]
            aspect_scale_xy, offset_padding_xy, bbox_lt_xy = required_data
            'expand_dims'
            if aspect_scale_xy.shape.rank==1:
               aspect_scale_xy, offset_padding_xy, bbox_lt_xy = self.ops_expand_batch_dim(
                   aspect_scale_xy, offset_padding_xy, bbox_lt_xy
                )
            'transform to  src img frame'
            kps_xy = (coords_xy-offset_padding_xy[:,None,:])*aspect_scale_xy[:,None,:] + bbox_lt_xy[:,None,:] #(1, 17,2) 
            kps_xy = tf.where(mask[...,None], kps_xy, 0.) #(1)
        else:
            raise ValueError(
                "miss required_keys ['aspect_scale_xy', 'offset_padding_xy', 'bbox_lt_xy'] \n"
                f" in data_to_src_img_frame : {data_to_src_img_frame.keys()}"
            ) 
        return kps_xy
        
    def batch_decode(
            self, 
            data : Dict[str,Tensor], 
            meta_data : Optional[dict] = None, *args, **kwargs
    ) -> tf.Tensor:
        r"""
        to_src_image = dict(
            aspect_scale_xy = tf.constant([1.,1.], dtype=self.compute_dtype),
            offset_padding_xy = tf.constant([0.,0.], dtype=self.compute_dtype),
            bbox_lt_xy= tf.constant([0.,0.], dtype=self.compute_dtype)
        )
        
        """

        batch_heatmaps = data['y_pred']

        if batch_heatmaps.shape.rank == 3:
           batch_heatmaps = tf.expand_dims(batch_heatmaps, axis=0) 
        elif  batch_heatmaps.shape.rank == 4:
            pass
        else:
            raise ValueError(
            )
        
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
            'concat src_ksp and score of heatmap'
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
            vis_fn : Optional[callable] =  None,
            take_batch_num : int = 1,
            batch_ids :  Union[List[int], int] = [0],  
            figsize : Tuple[int] = (16, 8),
            show_decoder_res : bool = True
        ):

        if vis_fn is None :
            vis_fn =  Vis_TopdownPoseHeatmapCodec(
                figsize =figsize,
                sel_batch_ids =batch_ids
            )
    
        else:
            if not isinstance(vis_fn,BasePoseVisFun):
                raise TypeError(
                    f"vis_fn must be 'BasePoseVisFun' type {self.__class__.__name__}"
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
class Vis_TopdownPoseHeatmapCodec(BasePoseVisFun):
    VERSION = '1.0.0'

    R"""  Vis_TopdownPoseSimCCLabelCodec
    Author : Dr. David Shen
    Date : 2024/3/29
    i.e. :


    """
    def __init__(
            self, **kwargs
    ):
        super().__init__(**kwargs)
        #self.simicc_split_xy_dim = simicc_split_xy_dim
        
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
        plot_kpt_pred = True  if decode_pred is not None and plot_kpt_true else False 
        plot_src_img = True if np_src_img is not None else False

        base_sub_plots = sum([plot_hm_true,plot_hm_pred,plot_kpt_true,plot_kpt_pred, plot_src_img])
        if plot_src_img : base_sub_plots += 1

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
        '#2 gt_heatmap'
        if plot_hm_true :
            idx += 1
            self.heatmaps_plot(
                y_trues = encode_true, 
                sum_axis= -1, 
                title = 'heatmap_true', 
                subplot_spec=(1,base_sub_plots,idx),           
            )  
        '#3 resized img with pred_kpts'
        if plot_kpt_pred :
            idx += 1
            self.base_plot(
                title = f"Resized Image with pred_kpts",
                image = np_resized_img,
                kpts = decode_pred,
                bboxes = None,
                subplot_spec = (1,base_sub_plots,idx)
            )
        '#4 pred_heatmap'
        if plot_hm_pred :
            idx += 1
            self.heatmaps_plot(
                y_trues = y_pred, 
                sum_axis= -1, 
                title = 'heatmap_pred', 
                subplot_spec=(1,base_sub_plots,idx),           
            )  
        '#4 src_img'         
        if plot_src_img:
            idx += 1
            #extra_text = f" ---(img_id : {image_id}, id : {instance_id} )"
            extra_text = f"\n( imgae_id : {image_id}, id : {instance_id} )"
            self.base_plot(
                title = f"Src Image" + extra_text,
                image = np_src_img,
                kpts = src_kpts,
                bboxes = src_bboxes,
                subplot_spec = (1,3,3)

            )   





