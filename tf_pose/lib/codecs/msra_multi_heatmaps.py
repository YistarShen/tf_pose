from typing import Dict, List, Optional, Tuple, Union, Sequence
from tensorflow import Tensor
import tensorflow as tf
#from .base import BaseKeypointCodec
from lib.Registers import CODECS
from .base_codec import BaseCodec
#---------------------------------------------------------------------
#
#----------------------------------------------------------------------
@CODECS.register_module()
class Multi_MSRAHeatmapCodec(BaseCodec):
    VERSION = '1.0.0'
    ENCODER_USE_PRED = False
    ENCODER_GEN_SAMPLE_WEIGHT = True
    r""" MSRAHeatmap used in heatmap base 2D pose detection,i.e. ViTPsoe/HRNet

          
    """
    
    def __init__(self, 
            use_udp : bool = True,
            num_kps : int =17, 
            sigmas : List[int] = [2, 3, 4, 5],
            target_size_xy : Tuple[int, int] = (192,256), 
            heatmap_size_xy : Tuple[int, int] = (48, 64),
            stack_multi_hm : bool = True,
            hm_thr : Union[tf.Tensor, List[float]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.use_udp = use_udp
        self.num_kps = num_kps
        self.num_units = len(sigmas)


        self.heatmap_size_xy = heatmap_size_xy
        self.target_size_xy = target_size_xy 
        self.sigmas = tf.cast(sigmas, dtype=tf.int32)
        
        self.stack_outputs = stack_multi_hm

        'enable UDP or no disable'
        if use_udp :
            self.feat_stride = ( tf.cast(target_size_xy,dtype=tf.float32)-1. ) / ( tf.cast(heatmap_size_xy,dtype=tf.float32)-1.)
        else:
            self.feat_stride = ( tf.cast(target_size_xy,dtype=tf.float32)) / ( tf.cast(heatmap_size_xy,dtype=tf.float32))
        
        'set hm_thr to decode'
        if isinstance(hm_thr, (Sequence, tf.Tensor)):
            self.hm_thr = tf.constant(hm_thr, dtype=tf.float32)
            assert  self.hm_thr.shape[0] == num_kps, \
             f"hm_thr.shape[0] must be equal to {num_kps}, but got {self.hm_thr.shape[0]} @MSRAHeatmap"  
        else:
            self.hm_thr = tf.zeros(shape=(num_kps,), dtype=tf.float32)                   
    
        
        'basic grid_map and core_xy'
        def multi_gaussian_kernel_gen(sigmas = [2, 3, 4, 5]):
            """Generate gaussian distribution with center value equals to 1."""
            grid_xy_list = []
            core_xy_list = []
            for  sigma in sigmas:
                heat_range = 2 *sigma*3+ 1
                x_core = y_core = heat_range // 2
                grid_coords_y = tf.cast(tf.range(0, heat_range), dtype=tf.float32)   #(grid_coords_y,)
                grid_coords_x = tf.cast(tf.range(0, heat_range), dtype=tf.float32)   #(grid_coords_x,)
                grid_x, grid_y = tf.meshgrid(grid_coords_x, grid_coords_y)  # (grid_coords_x,grid_height)
                grid_x = tf.expand_dims(grid_x, axis=-1)   # (grid_coords_x,grid_height,1)
                grid_y = tf.expand_dims(grid_y, axis=-1)
                grid_xy = tf.concat([grid_x, grid_y], axis=-1)  # (13,13,2)
                grid_xy_list.append(grid_xy)
                core_xy_list.append(tf.constant([x_core,y_core], dtype=tf.float32))
            return  grid_xy_list, core_xy_list

        sigmas_square_list = [  tf.math.square(tf.cast(sigma,dtype=tf.float32)) for sigma in sigmas]   #????
        grid_maps_list, cores_xy_list = multi_gaussian_kernel_gen(sigmas) 


        # self.grid_map_branch_fns = {0: lambda: (grid_maps_list[0], cores_xy_list[0], sigmas_square_list[0]), 
        #                             1: lambda: (grid_maps_list[1], cores_xy_list[1], sigmas_square_list[1]), 
        #                             2: lambda: (grid_maps_list[2], cores_xy_list[2], sigmas_square_list[2]), 
        #                             3: lambda: (grid_maps_list[3], cores_xy_list[3], sigmas_square_list[3])}
        
        self.unit_branch_fns = dict()
        lambdas = [lambda i=i: (grid_maps_list[i], cores_xy_list[i], sigmas_square_list[i]) 
                    for i in range(self.num_units)]
        for i , lambda_fn in enumerate(lambdas) :
            self.unit_branch_fns.update({i : lambda_fn })

        
        'set cfg for tf.map_fn'
        if not self.use_vectorized_map:
            self.HeatmapTensorSpec =  tf.type_spec_from_value(
                tf.ones(shape=(heatmap_size_xy[1],heatmap_size_xy[0]), dtype=tf.float32)
            )
            self.VisTensorSpec = tf.type_spec_from_value(
                tf.constant(1, dtype=tf.float32)
            ) 
            self.parallel_iterations *= self.num_units*self.num_kps

        
    def gen_heatmap(self, 
                    flatten_kps_dist, 
                    lu, rb, cond, indices):
        ''' 
        flatten_kps_dist : (2,)
        lu : (2,)
        lu : (2,) 
        cond : () 
        indices : () 
        '''

        def true_fn():
            heatmap = tf.zeros(shape=(self.heatmap_size_xy[1], self.heatmap_size_xy[0])) 
            target_weight = tf.constant(0, dtype=tf.float32)
            return heatmap, target_weight
        
        def false_fn():

            grid_xy, core_xy, sigma_square = tf.switch_case(
                tf.cast(indices,dtype=tf.int32), 
                branch_fns=self.unit_branch_fns, 
            )

            # unit_branch_fns / grid_map_branch_fns
            #flatten_kps_dist
            core_xy = core_xy + flatten_kps_dist 
            
            diff_square = tf.math.squared_difference( 
                grid_xy, core_xy, name=None
            )#(13,13,2)
            diff_square_sum = tf.reduce_sum(
                diff_square,axis=-1
            )#(13,13)
            gaussian_map = tf.math.exp( 
                -diff_square_sum/(2*sigma_square) 
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
            target_weight = tf.constant(1, dtype=tf.float32)

            return heatmap, target_weight
        

        heatmap, target_weight =  tf.switch_case(
            tf.cast(cond,dtype=tf.int32), 
            branch_fns={0: false_fn, 1: true_fn}, 
            default=false_fn
        )

        return {'heatmap': heatmap, 
                'target_weight' : target_weight}
    
        # if  self.embedded_codec :
        #     sample_weights = tf.stop_gradient(sample_weights)
        #     heatmaps = tf.stop_gradient(heatmaps)

    def gen_targets(self,kps_true):
        # kps_true : (b,17,3)

        'kps_true (b, 17, 2) & vis (b, 17, 1)'
        vis = kps_true[..., 2:3]
        kps_true = kps_true[...,:2]


        'hm_kps / hm_kps_int (b, 17, 2)'
        hm_kps = kps_true/self.feat_stride #(b,17,2)
        hm_kps_int = tf.cast( hm_kps + 0.5, dtype=tf.int32) # #(b,17,2)
        'kps_dist(b, 17, 4, 2)'
        kps_dist = (hm_kps[...,:2] - tf.cast(hm_kps_int[...,:2], dtype=tf.float32)) # (b,17,2)
        kps_dist =  tf.tile( kps_dist[...,None,:], (1,1,self.num_units,1))# (b,17,4, 2)
        'vis (b, 17, 4)'
        #vis =  kps_true[..., 2:3] #(b, 17, 1)
        vis = tf.tile(vis,(1, 1, self.num_units)) #(b,17,4)
        'unit_indices (b, 17, 4) @int'
        unit_indices = tf.ones_like(vis, dtype=tf.int32)*tf.range(0,self.num_units) #(b,17,4)
        'sigma (b, 17, 4) @int'
        sigmas = tf.ones_like(vis, dtype=tf.int32)*self.sigmas#(b, 17, 4)
        'lu/rb (b, 17, 4, 2)'
        lu = tf.cast( 
                hm_kps_int[...,None,:2] - sigmas[...,None]*3, dtype=tf.int32
        )#(b,17,4, 2)
        rb = tf.cast( 
                hm_kps_int[...,None,:2] + sigmas[...,None]*3 + 1, dtype=tf.int32
        )#(b,17,4, 2)


        'flatten_unit_indices  (b*17*4,)'
        flatten_unit_indices = tf.reshape(unit_indices, [-1]) # (b*17*4)
        'flatten_kps_dist  (b*17*4,2)'
        flatten_kps_dist = tf.reshape(kps_dist, [-1,2]) # (b*17*4,2)
        'flatten_sigmas  (b*17*4,2)'
        #flatten_sigmas = tf.reshape( sigmas, [-1]) #(b*17*4)
        'flatten lu /rb  (b*17*4,2)'
        lu =  tf.reshape(lu,[-1, 2]) # (b*17*4,2)
        rb =  tf.reshape(rb,[-1, 2]) # (b*17*4,2)
        'flatten vis  (b*17*4,1)'
        vis = tf.reshape(vis,[-1,1]) #(b*17*4,1)
        'flatten cond  (b*17*4,)' 
        cond_lu = tf.greater_equal(lu, self.heatmap_size_xy) #(b*17*4,2)
        cond_rb = tf.less( rb, 0) #(b*17*4,2)
        cond_vis = tf.equal(vis, 0.) #(b*17*4,1)
        cond = tf.concat([cond_lu,cond_rb, cond_vis], axis=-1) #(b*17*4,5)
        cond = tf.math.reduce_any(cond, axis=-1) #(b*17*4)


        if self.use_vectorized_map :
            flatten_res = tf.vectorized_map(
                lambda x : self.gen_heatmap(x[0], x[1], x[2], x[3], x[4]), 
                (flatten_kps_dist, lu, rb, cond, flatten_unit_indices) ,
                fallback_to_while_loop=True
            )#(b*17, 64, 48)
        else:
            flatten_res = tf.map_fn(
                lambda x : self.gen_heatmap(x[0], x[1], x[2], x[3], x[4]),
                (flatten_kps_dist, lu, rb, cond,  flatten_unit_indices) ,
                parallel_iterations = self.parallel_iterations*self.num_units*self.num_kps,
                fn_output_signature = {
                        'heatmap' : self.HeatmapTensorSpec ,
                        'target_weight' : self.VisTensorSpec,
                }
            )#(b*17, 64, 48)

        flatten_heatmaps = flatten_res['heatmap'] #(b*17*4, 64, 48)
        flatten_target_weight = flatten_res['target_weight'] #(b*17*4)
        multi_heatmaps = tf.reshape(
                flatten_heatmaps,
                [-1,self.num_kps, 4,self.heatmap_size_xy[1],self.heatmap_size_xy[0]]
        )
        multi_heatmaps = tf.transpose(multi_heatmaps,perm=[0,2,3,4,1]) #(b,4,64,48,17)
        # target_weights = tf.reshape(
        #         flatten_target_weight,
        #         [-1,self.num_kps, self.num_units,1]
        # )#(b,17,4,1)
        # target_weights = tf.transpose(target_weights,perm=[0,2,1,3]) #(b,4,17,1)
        # sample_weights = tf.concat(
        #         [tf.tile(kps_true[:, None,...],(1,self.num_units,1,1)), target_weights], axis=-1
        # )#(b,4,17,3)   


        target_weights = tf.reshape(
                flatten_target_weight,
                [-1,self.num_kps, self.num_units]
        )#(b,17,4)
        target_weights = tf.transpose(target_weights,perm=[0,2,1]) #(b,4,17)
        # sample_weights = tf.concat(
        #         [tf.tile(kps_true[:, None,...],(1,self.num_units,1,1)), target_weights], axis=-1
        # )#(b,4,17,3)      


        if  self.embedded_codec :
            sample_weights = tf.stop_gradient(target_weights)  #(b,4,17)
            multi_heatmaps = tf.stop_gradient(multi_heatmaps)  #(b,4,64,48,17)
        

        return multi_heatmaps, sample_weights
    
    def batch_encode(self,
                    data : Dict[str,tf.Tensor],
                    y_pred : Optional[ Union[Tuple[Tensor],Tensor]]=None) -> Dict[str,tf.Tensor]:
        

        kps_true = data['kps'] #(b,17,3)
        heatmaps, sample_weights = self.gen_targets(kps_true)
        if not self.stack_outputs:
            multi_heatmaps = [tf.squeeze(feat, axis=1) 
                              for feat in tf.split(multi_heatmaps, self.num_units, axis=1)
            ]
            sample_weights = [tf.squeeze(feat, axis=1) 
                              for feat in tf.split(sample_weights, self.num_units, axis=1)
            ]

        data['y_true'] = heatmaps#(b,4,17,3)
        data['sample_weight'] = sample_weights#(b,4,17)
        #data['sample_weight'] = sample_weights
        return data
    
# #---------------------------------------------------------------------
# #
# #----------------------------------------------------------------------
# #----------------------------------------------------------------------
# @CODECS.register_module()
# class MultiHeatmaps(BaseKeypointCodec):
#     r""" MSRAHeatmap used in heatmap base 2D pose detection,i.e. ViTPsoe/HRNet
#         Input kps : (x, y, vis)
#         Output heatmap_size : (w,h) or (x,y)

#     SimCC_PCKMetric is used in training progress to monitor cuurent accuracy of pose estimation model
  
#     Args:
#         use_udp (bool): 
#         num_kps (int): 
#         radius_list (List[int]) :
#         target_size_xy (tuple) :
#         heatmap_size_xy (tuple) :
#         stack_multi_hm (bool) :
#         hm_thr (List[float] | tf.tensor):  threshold value of heatmap to get effective kps

#     batch_decode :
         

#     Example :

#     Example :
#         - coco_17kps_256x192_cfg = dict(
#                                         use_udp = True, 
#                                         num_kps = 17,
#                                         radius_list = [7, 9, 11, 15]
#                                         target_size_xy = (192,256),
#                                         heatmap_size_xy = (48, 64),
#                                         stack_multi_hm  = False,
#                                         hm_thr= [0.75, 0.75, 0.75, 0.75, 0.75,
#                                                 0.25, 0.25, 0.25, 0.25, 0.25,
#                                                 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
#         )

          
#     """
    
#     def __init__(self, 
#             use_udp : bool = True,
#             num_kps : int =17, 
#             radius_list : List[int] = [7, 9, 11, 15],
#             target_size_xy : Tuple[int, int] = (192,256), 
#             heatmap_size_xy : Tuple[int, int] = (48, 64),
#             stack_multi_hm : bool = False,
#             hm_thr : Union[tf.Tensor, List[float]] = None) ->None:
#         """
#         sigma : 2
#         sigma : [5, 9, 11, 15]
#                 2x3-1, 2x4
#         """
    
#         self.use_udp = use_udp
#         self.num_kps = num_kps
#         self.heatmap_size_xy = heatmap_size_xy
#         self.target_size_xy = target_size_xy 
#         self.stack_outputs = stack_multi_hm

#         radius_list  #(rule : sigma*3 = radius )
#         sigma_square_list = [tf.math.square(tf.cast(radius/3, dtype=tf.float32)) for radius in radius_list]


#         'enable UDP or no disable'
#         if use_udp :
#             self.feat_stride = ( tf.cast(target_size_xy,dtype=tf.float32)-1. ) / ( tf.cast(heatmap_size_xy,dtype=tf.float32)-1.)
#         else:
#             self.feat_stride = ( tf.cast(target_size_xy,dtype=tf.float32)) / ( tf.cast(heatmap_size_xy,dtype=tf.float32))
        
#         'set hm_thr to decode'
#         if hm_thr is None :
#             self.hm_thr = tf.zeros(shape=(num_kps,), dtype=tf.float32)
#         else:
#             if isinstance(hm_thr,tf.Tensor): 
#                 self.hm_thr = hm_thr
#             elif isinstance(hm_thr,List): 
#                 self.hm_thr = tf.constant(hm_thr, dtype=tf.float32)
#             else:
#                 raise RuntimeError(f"type of hm_thr must be tf.Tensor|List,but got {type(hm_thr)}") 

#             assert  self.hm_thr.shape[0] == num_kps, \
#              f"hm_thr.shape[0] must be equal to {num_kps}, but got {self.hm_thr.shape[0]} @MSRAHeatmap"                       
    
        
#         'basic grid_map and core_xy'
#         def multi_gaussian_kernel_gen(sigmax3_list = [7, 9, 11, 15]):
#             """Generate gaussian distribution with center value equals to 1."""
  
#             grid_xy_list = []
#             core_xy_list = []
#             for  sigma_x3 in sigmax3_list:
#                 #heat_range = 2 * sigma * 3 + 1
#                 heat_range = 2 *sigma_x3+ 1
#                 x_core = y_core = heat_range // 2
#                 grid_coords_y = tf.cast(tf.range(0, heat_range), dtype=tf.float32)   #(grid_coords_y,)
#                 grid_coords_x = tf.cast(tf.range(0, heat_range), dtype=tf.float32)   #(grid_coords_x,)
#                 grid_x, grid_y = tf.meshgrid(grid_coords_x, grid_coords_y)  # (grid_coords_x,grid_height)
#                 grid_x = tf.expand_dims(grid_x, axis=-1)   # (grid_coords_x,grid_height,1)
#                 grid_y = tf.expand_dims(grid_y, axis=-1)
#                 grid_xy = tf.concat([grid_x, grid_y], axis=-1)  # (13,13,2)
#                 grid_xy_list.append(grid_xy)
#                 core_xy_list.append(tf.constant([x_core,y_core], dtype=tf.float32))
#             return  grid_xy_list, core_xy_list
            
#         grid_map_list, core_xy_list = multi_gaussian_kernel_gen(radius_list) 
#         self.num_features = len(grid_map_list)
#         assert self.num_features==len(core_xy_list)==len(radius_list)==len(sigma_square_list)

#         self.multi_hm_cfg =  (grid_map_list, core_xy_list, radius_list, sigma_square_list)

#     def batch_encode(self,
#                     data : Dict[str,tf.Tensor],
#                     use_vectorized_map :bool=False) -> Dict[str,tf.Tensor]:
#         'add new keys for encode results'
#         # data['y_true'] = tf.zeros(shape=(self.target_size_xy[1],self.target_size_xy[0], 3), dtype=tf.float32)
#         # data['sample_weight'] = tf.zeros(shape=(self.num_kps,3), dtype=tf.float32)
#         # add new keys : Shape () must have rank at least 1
#         data['y_true'] = tf.zeros(shape=(1,))
#         data['sample_weight'] = tf.zeros(shape=(1,))
#         if use_vectorized_map:
            
#             data = tf.vectorized_map(self.encode, 
#                                     data, 
#                                     fallback_to_while_loop=True, 
#                                     warn=True)
#             return data
#         else:
#             data = tf.map_fn(self.encode,
#                              data
#             )
#         return data
    

#     def encode(self,
#                data : Dict[str,tf.Tensor]) -> Dict[str,tf.Tensor]:
             
#         multi_sample_weights_list = []
#         multi_heatmaps_list = []

#         kps_true = data['kps']
#         hm_kps = kps_true[...,:2]/self.feat_stride #(17,2)
#         hm_kps_int = tf.cast( hm_kps + 0.5, dtype=tf.int32) # to check

#         for grid_map, core_xy, radius, sigma_square in zip(self.multi_hm_cfg):
#             heatmaps = []
#             target_weights = []
#             for j in range (self.num_kps):
#                 lu = tf.cast( hm_kps_int[j,:2] - radius, dtype=tf.int32) 
#                 rb = tf.cast( hm_kps_int[j,:2] + radius + 1, dtype=tf.int32)
#                 if (lu[0] >= self.heatmap_size_xy[0] or lu[1] >= self.heatmap_size_xy[1] or rb[0] < 0 or rb[1] < 0 or kps_true[j, 2]==0):
#                     heatmap = tf.zeros(shape=(self.heatmap_size_xy[1], self.heatmap_size_xy[0],1)) 
#                     target_weight = tf.constant(0, dtype=tf.int32)
#                 else:
#                     core_xy = core_xy + ( hm_kps[j,:2] - tf.cast(hm_kps_int[j,:2], dtype=tf.float32)) #(2,)
#                     diff_square = tf.math.squared_difference( grid_map, core_xy, name=None)
#                     diff_square_sum = tf.reduce_sum(diff_square,axis=-1)
#                     gaussian_map = tf.math.exp( -diff_square_sum/(2*sigma_square) )
#                     'get effective region'
#                     x_gauss_left = tf.math.maximum(0, -lu[0])
#                     x_gauss_right = tf.math.minimum(rb[0], self.heatmap_size_xy[0]) - lu[0]
#                     y_gauss_up = tf.math.maximum(0, -lu[1])
#                     y_gauss_bottom = tf.math.minimum(rb[1], self.heatmap_size_xy[1]) - lu[1]  
                
#                     eff_gaussian_map = gaussian_map[y_gauss_up:y_gauss_bottom, x_gauss_left:x_gauss_right]
            
#                     paddings = tf.stack([ tf.reverse(lu, axis=[0]), tf.reverse(self.heatmap_size_xy-rb, axis=[0])],  axis=-1)
#                     paddings = tf.math.maximum([[0, 0], [0, 0]], paddings )
#                     heatmap = tf.pad(eff_gaussian_map, paddings)
#                     heatmap = tf.expand_dims(heatmap, axis=-1)
#                     target_weight = tf.constant(1, dtype=tf.int32)

#                 heatmaps.append(heatmap)
#                 target_weights.append(target_weight)
            
#             heatmaps = tf.concat(heatmaps, axis=-1)
#             target_weights = tf.cast(tf.stack(target_weights, axis=0), dtype=tf.float32)    
#             sample_weights = tf.concat([kps_true[:, :2], tf.expand_dims(target_weights, axis=-1)], axis=-1)

#             'output list (y_true : multi_heatmaps_list, sample_weight : multi_target_weights_list'
#             multi_heatmaps_list.append(heatmaps)
#             multi_sample_weights_list.append( sample_weights )
       
  
#         # multi_heatmaps  = tf.stack(multi_heatmaps_list, axis=0)             #(4,17,3)
#         # multi_sample_weights  = tf.stack(multi_sample_weights_list, axis=0) #(4,17,3)
#         # data['y_true'] = multi_heatmaps
#         # data['sample_weight'] = multi_sample_weights

#         data['y_true'] = tf.stack(multi_heatmaps_list, axis=0)  if self.stack_outputs else multi_heatmaps_list
#         data['sample_weight'] = tf.stack(multi_sample_weights_list, axis=0) if self.stack_outputs else multi_sample_weights_list
#         return data
    
#     def transform_preds(self,
#                     coords_xy, 
#                     mask,
#                     aspect_scale_xy,
#                     offset_padding_xy, 
#                     bbox_lt_xy):
#         #assert coords.shape[1] in (2, 4, 5)
#         assert aspect_scale_xy.shape[1] == 2, f"{aspect_scale_xy.shape[1]}"
#         assert offset_padding_xy.shape[1] == 2, f"{offset_padding_xy.shape[1]}"
#         assert bbox_lt_xy.shape[1] == 2, f"{bbox_lt_xy.shape[1]}"
#         assert coords_xy.shape[1] == mask.shape[1], f"{coords_xy.shape[1]} and {mask.shape[1]}"
#         assert coords_xy.shape.rank == 3, f"{coords_xy.shape.rank }"
#         assert mask.shape.rank == 2, f"{mask.shape.rank}"
 
#         kps_xy = (coords_xy-offset_padding_xy[:,None,:])*aspect_scale_xy[:,None,:] + bbox_lt_xy[:,None,:] #(1, 17,2) 
#         kps_xy = tf.where(mask[...,None], kps_xy, 0.) #(1)

#         return kps_xy

#     def decode(self, 
#             multi_heatmaps : tf.Tensor, 
#             kernel : int=3,
#             normalize : bool = False,
#             src_img_frame : bool = False,                   
#             aspect_scale_xy = tf.constant([1.,1.], dtype=tf.float32),
#             offset_padding_xy = tf.constant([0.,0.], dtype=tf.float32),
#             bbox_lt_xy= tf.constant([0.,0.], dtype=tf.float32)) -> tf.Tensor:
        
#         if isinstance(multi_heatmaps, (list, tuple)) :
#             heatmaps = multi_heatmaps[0] #[b,hm_nums,h,w,kps_nums]
    
#         if isinstance(multi_heatmaps, tf.Tensor) and multi_heatmaps.shape.rank == 5:
#             heatmaps = multi_heatmaps[:,0,...]
             

#         expand_batch_dim = True if heatmaps.shape.rank == 3 else False

#         if expand_batch_dim:
#             batch_heatmaps = tf.expand_dims(heatmaps, axis=0) 
#         else:
#             batch_heatmaps = heatmaps

#         batch_keypoints, mask = self.batch_decode(batch_heatmaps,
#                                                 kernel, 
#                                                 normalize)
        
        
#         if src_img_frame:
            
#             if aspect_scale_xy.shape.rank == 1:
#                 aspect_scale_xy = tf.expand_dims(aspect_scale_xy, axis=0)

#             if offset_padding_xy.shape.rank == 1:
#                 offset_padding_xy = tf.expand_dims(offset_padding_xy, axis=0)

#             if bbox_lt_xy.shape.rank == 1:
#                 bbox_lt_xy = tf.expand_dims(bbox_lt_xy, axis=0)

#             'get kps of src frame'
#             kps_xy = self.transform_preds(batch_keypoints[...,:2], 
#                                     mask, 
#                                     aspect_scale_xy, 
#                                     offset_padding_xy, 
#                                     bbox_lt_xy)
    
#             'concat src_ksp and socre of heatmap'
#             batch_keypoints = tf.concat([kps_xy, batch_keypoints[:,:,2:]], axis=-1) 

#         if expand_batch_dim:
#             return tf.squeeze(batch_keypoints, axis=0)
        
#         return batch_keypoints
    
#     #@tf.function
#     def batch_decode(self, 
#                     batch_heatmaps : tf.Tensor, 
#                     kernel : int=3,
#                     normalize : bool = False ) -> tf.Tensor:
        

#         batch, height, width, joints = batch_heatmaps.shape
        
#         assert height==self.heatmap_size_xy[1] and width==self.heatmap_size_xy[0], \
#         f"heatmaps_wh to decode is {width}x{height}, it didn't match {self.heatmap_size_xy} @MSRAHeatmap.batch_decode"
        

#         assert ( joints==self.num_kps),f"joints_num must be {self.num_kps}, but got {joints} @MSRAHeatmap.batch_decode"

#         def nms(heat, kernel=5):
#             hmax = tf.nn.max_pool2d(heat, kernel, 1, padding='SAME')
#             keep = tf.cast(tf.equal(heat, hmax), tf.float32)
#             return heat*keep
        
#         batch_heatmaps = nms(batch_heatmaps, kernel=kernel ) #(1,96,96,14)
#         flat_tensor = tf.reshape(batch_heatmaps, [tf.shape(batch_heatmaps)[0], -1, tf.shape(batch_heatmaps)[-1]]) #(1,96*96,14)
#         'Argmax of the flat tensor'
#         argmax = tf.argmax(flat_tensor, axis=1) #(1,14)
#         argmax = tf.cast(argmax, tf.int32) #(1,14)
#         scores = tf.math.reduce_max(flat_tensor, axis=1) #(1,14)
#         'Convert indexes into 2D coordinates'
#         argmax_y = tf.cast( (argmax//width), tf.float32)  #(1,14)
#         argmax_x = tf.cast( (argmax%width), tf.float32)   #(1,14)

#         if normalize:
#             argmax_x = argmax_x / tf.cast(width, tf.float32)
#             argmax_y = argmax_y / tf.cast(height, tf.float32)

#         # Shape: batch * 3 * n_points  (1,14,3)
#         batch_keypoints = tf.stack((argmax_x, argmax_y, scores), axis=2)
#         'mask'
#         mask = tf.greater(batch_keypoints[..., 2], self.hm_thr[None,:]) #(b,17)

#         kps_xy = tf.where(mask[:,:,None], batch_keypoints[:, :, :2], 0)*self.feat_stride
#         batch_keypoints = tf.concat([kps_xy, scores[:,:,None]], axis=-1) 

#         return batch_keypoints, mask
    


