from typing import Dict, List, Optional, Tuple, Union, Any, Sequence, Callable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from lib.datasets.transforms.utils import PackInputTensorTypeSpec
class BasePoseVisFun:
    VERSION = '1.1.0'
    
    r""" BasePoseVisFun
        Author : Dr. David Shen
        Date : 2024/3/20
    


    """
    def __init__(
            self, 
            figsize : Tuple[int] = (24, 8),
            sel_batch_ids : Optional[List[int]] = [0],
            plot_transformed_bbox : bool = True
    ):
        self.figsize = figsize
        self.sel_batch_ids = sel_batch_ids
        self.plot_transformed_bbox = plot_transformed_bbox
        self.num_batched_samples = None
 
    def parse_subplot_spec(self, subplot_spec, **kwargs):
        if type(subplot_spec)==tuple:
            plt.subplot(*subplot_spec,**kwargs)
        else:
            plt.subplot(subplot_spec, **kwargs)


    def base_plot(
        self, 
        image, 
        bboxes=None, 
        labels=None, 
        kpts=None, 
        title='Image',
        subplot_spec=(1,1,1),
        **kwargs
    ):  
        
        self.parse_subplot_spec(
            subplot_spec=subplot_spec, **kwargs
        )
        plt.title(
            title, fontsize = 12
        )
        plt.imshow(image)

        is_multi_poses = (bboxes is not None and bboxes.shape.rank==2)

        num_obj = bboxes.shape[0] if is_multi_poses else 1
        ith_obj = -1 
        for i in range(num_obj):   
            label = labels[i] if labels is not None else 1
            
            if label==1: 
                color = np.random.uniform(
                    low=0., high=1., size=3
                )
                ith_obj +=1
            #print('bbox2',labels, bboxes.shape, num_obj, ith_obj)

            if bboxes is not None :   
                bbox = bboxes[i] if is_multi_poses else bboxes
                #print('bbox',bbox.shape)
                x1, y1, w, h = bbox
                ax = plt.gca() 
                patch = plt.Rectangle(
                    [x1, y1], w, h, 
                    fill=False, 
                    edgecolor=color, 
                    linewidth=2
                )
                ax.add_patch(patch)
                text = "{}".format(label)
                ax.text(
                    x1, y1, text,
                    bbox={"facecolor": color, "alpha": 0.8},
                    clip_box=ax.clipbox,
                    clip_on=True,
                )
                

            if kpts is not  None and label == 1 :
                kpt_ith = kpts[ith_obj] if is_multi_poses else kpts
                for j in range(0,kpt_ith.shape[0]):
                    kps_x = int((kpt_ith[j,0]))
                    kps_y = int((kpt_ith[j,1]))
                    plt.scatter(kps_x,kps_y, color=color)

    def heatmaps_plot(
            self,  
            y_trues : Tensor, 
            sum_axis= -1, 
            title : str='heatmap', 
            subplot_spec=(1,1,1),
            **kwargs
    ):
        self.parse_subplot_spec(subplot_spec=subplot_spec)
        plt.title(title, fontsize = 12)
        heatmaps = y_trues.numpy()
        plt.imshow(heatmaps.sum(axis=sum_axis))


    def xy_heatmaps_plot(
        self, 
        gs : gridspec.GridSpec,
        x_heatmaps : Tensor, 
        y_heatmaps : Tensor, 
        lt_start_gxy : Tuple[int] = (0,0), 
        title = "simcc_",
        hm_1d_size : int = 5,
        hm_1d_intervel_row : int = 5,
        hm_1d_intervel_col : int = 5,
    ): 
        if not isinstance(gs,gridspec.GridSpec):
            raise TypeError(
                f"input 'gs' must be  matplotlib.gridspec.GridSpec type in xy_heatmaps_plot() "
                f"but got {type(gs)} @ {self.__class__.__name__} "
            ) 
        num_joints = y_heatmaps.shape[0]
        lt_start_gx, lt_start_gy = lt_start_gxy
        heatmaps_2d = x_heatmaps[:,None,:]*y_heatmaps[:,:,None] #(17,simcc_y, simcc_x)
        hm_2d_size_hw = heatmaps_2d.shape[1:] #(512,384)
        heatmaps_2d = tf.reduce_max(heatmaps_2d, axis=0).numpy()
       
        '2d heatmap'
        plt.subplot(
            gs[lt_start_gy:lt_start_gy + hm_2d_size_hw[0]//2, lt_start_gx:lt_start_gx+hm_2d_size_hw[1]//2]
        )
        plt.title(
            title + "Heatmap2D", fontsize = 12
        )
        plt.imshow(heatmaps_2d)
        '1d heatmap'
        start_gy = lt_start_gy + hm_2d_size_hw[0]//2+hm_1d_intervel_row+10
        start_gx = lt_start_gx  + hm_2d_size_hw[1]//2+hm_1d_intervel_col
        
        for i in range(num_joints) :
            x_heatmap = x_heatmaps[i][None,:]
            ax2 = plt.subplot(gs[start_gy:start_gy+hm_1d_size, lt_start_gx:lt_start_gx+hm_2d_size_hw[1]//2])
            plt.imshow(x_heatmap, aspect='auto')
            ax2.set_yticks([])
            if i!=num_joints-1 :
                ax2.set_xticks([])
            ax2.set_ylabel(f'{i}',fontweight ='bold', fontsize=7).set_color('red')
            start_gy += hm_1d_intervel_row+hm_1d_size


            y_heatmap = y_heatmaps[i][:, None] 
            ax2 = plt.subplot(gs[ lt_start_gy: lt_start_gy + hm_2d_size_hw[0]//2, start_gx:start_gx+hm_1d_size])
            plt.imshow(y_heatmap, aspect='auto')
            if i!=num_joints-1 :
                ax2.set_yticks([])
            else:
                ax2.yaxis.set_label_position("right")
                ax2.yaxis.tick_right()
            ax2.set_xticks([])
            ax2.set_xlabel(f'{i}',fontweight ='bold', fontsize=7).set_color('red') 
            start_gx += hm_1d_intervel_col+hm_1d_size  



    def map_data_structure(
        self, data : dict
    )->tuple:
        
        num_batched_samples = None
        batch_size = 0

        for key, val in data.items():

            if val==None:
               continue  
            if val==[]:
               data[key] = None
               continue 

            if isinstance(val, Tensor):
                val = [ val ]

            if isinstance(val, list) : 
                'get and verify batch_size'
                if not batch_size :
                    batch_size =val[0].shape[0] 
                    #print(f"batch_size : {batch_size}")
                else:
                    #print(key, batch_size, type(val[0]), val[0].shape)
                    assert(val[0].shape[0]==batch_size) 
                
                data[key] =  tf.stack(
                    val, axis=0
                )
                'get and verify num_batched_samples'
                if not num_batched_samples :
                    num_batched_samples = data[key].shape[0]
                else:
                    
                    assert(data[key].shape[0]==num_batched_samples) 

            else:
                raise TypeError(
                    f""
                )  
        """ 
        data : dict(
                    key1 = (num_batched_samples, batch_size, ....),
                    key2 = (num_batched_samples, batch_size, ....),
                    key2 = (num_batched_samples, batch_size, ....),
                )
        """       
        return data , num_batched_samples  
    
    def all_vaild(self,data_list : list):
        return all(v is not None for v in data_list) 

    def extract_key_samples(
        self, samples_list : List[Dict[str,Tensor]], key : str
    ) : 
        if not all(samples_list):
            return None
        data = [ samples.get(key, None)for samples in samples_list]
        return data if self.all_vaild(data) else None

    def call(
        self,  **kwargs
    ):
        r"""
        args :
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
        
 
        """
        raise NotImplementedError()


    def parse_dataset(
            self, batched_samples_list : List[Dict[str,Tensor]] = [],  
        ) :
        r""" Example 
        
        """
        if type(batched_samples_list[0])!= dict :
            raise TypeError(
                f""
            ) 
        raise NotImplementedError()
        

    def __call__(
        self, 
        batched_samples_list : List[Dict[str,Tensor]] = [],  
        show_type_sapec : bool = True,
        sel_batch_ids : Optional[List[int]] = None,
        plot_transformed_bbox : Optional[bool] = None,
        **kwargs
    )-> None :
        
        R""" kwargs : 
                y_preds : Optional[List[Tensor]] = None,
                decoded_preds : Optional[List[Tensor]] = None,
                encoded_trues : Optional[List[Tensor]] = None,
                kpts : Optional[List[Tensor]] = None,
                bboxes : Optional[List[Tensor]] = None,
                labels : Optional[List[Tensor]] = None,
                src_imgs : Optional[List[Tensor]] = None,
                src_kpts : Optional[List[Tensor]] = None,
                src_bboxes : Optional[List[Tensor]] = None,
                src_labels : Optional[List[Tensor]] = None,
                src_img_ids : Optional[List[Tensor]] = None,
                src_ids : Optional[List[Tensor]] = None,  
        """
        if isinstance(sel_batch_ids, list) and len(sel_batch_ids)!=0:   
            self.sel_batch_ids = sel_batch_ids
        
        if plot_transformed_bbox is not None :
            self.plot_transformed_bbox = plot_transformed_bbox

        data = dict()
        if isinstance(batched_samples_list, list) and len(batched_samples_list)!=0:
            if type(batched_samples_list[0])!= dict :
                raise TypeError(
                    f"the element in batched_samples_list must be 'dict' type to parse"
                ) 
            data  =  self.parse_dataset(batched_samples_list)

        data = {**data,**kwargs}
        
        if all( [(val==None or val==[]) for val in data.values()]):
                raise ValueError(
                    f"all input values are None or [], please check input data"
            ) 

        
        data , num_batched_samples  = self.map_data_structure(data)



        for i in range(num_batched_samples):
            for j, batch_id in enumerate(self.sel_batch_ids):
                'TensorTypeSpec of data'
                if show_type_sapec and i==0 and j==0:
                    PackInputTensorTypeSpec(
                        tf.nest.map_structure(lambda x : x[0,...] if x is not None else None, data ) ,{}, show_log=True
                    )
                
                sel_data = tf.nest.map_structure(
                    lambda x : x[i, batch_id,...] if x is not None else None, data     
                ) 
                'sel_data (single one sample) for ploting'
                self.call(**sel_data)    
        #return  data  