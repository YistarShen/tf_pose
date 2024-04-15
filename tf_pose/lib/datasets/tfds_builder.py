from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import copy, warnings
import tensorflow as tf
from tensorflow import Tensor
from engine.tfds_pipeline.base import tfds_Base
from lib.codecs import BaseCodec
from lib.Registers import DATASETS, CODECS
from lib.datasets.transforms import PackInputTensorTypeSpec

############################################################################
#
# 
############################################################################
@DATASETS.register_module()
class dataloader(tfds_Base):
    VERSION = '2.0.2'
    r"""dataloader to  gerenrate tf.data.Dataset by paser, agument pipeline and codec
    Date : 2024/2/22
    Author : Dr. David Shen 

   
    1. from VERSION = 1.0.1, dataloader can support multi-datasets, 
    2. from VERSION = 2.0.0, dataloader becomes a common function for pose and det datasets



    data Preprocessing composed of 
                    -1. prepare_data 
                    -2. augments : data transforms 
                    -3. encoder : generate tragets for mdoel training
    
    
    Args:
        tfrec_dataset (List[dict]):  parse specific dataset and get its total smaples
        codec (dict):  encode dataset such as kps to heatmaps and decode encoded data
        batch_size (int): batch size of dataset
        prefetch_size (int): prefetch_size on batch size
        shuffle (bool) : whether shuffle dataset and shuffle size depends on batch size
        augmenters (list): data augments used in dataset


    i.e. :
       tfrec_dataset = [ dict(type = 'BaseCocoStyleDataset_Parser', data_root = coco_tfrec_data_dir) 
                         dict(type = 'AIC_CocoStyleDataset_Parser', data_root = aic_tfrec_data_dir)]

       codec = dict(type='UDP_SimCCLabel',
            sigma_xy = (4.9,5.66), 
            image_size_xy = IMG_SIZE, 
            simcc_split_ratio = float(SIMCC_SPLIT_RATIO),
            normalize = False,
            use_udp = True)

        pipeline = [
                    dict(type='RandomFlip', direction='horizontal'),
                    dict(type='RandomHalfBody'),
                    dict(type='TopdownAffine', input_size=codec['input_size']),
        ]


    """

    def __init__(
            self,
            tfrec_datasets_list : List[Dict],
            codec : Optional[dict] = None,
            batch_size : Optional[int]=256,
            prefetch_size : Optional[int]=4,
            shuffle  :  Optional[bool]=True,
            ensure_to_tensor : bool = False,
            augmenters: List[Union[dict, Callable]] = [],
            parallel_iterations : int = 16,
            vis_fn : Callable = None
    ): 
        self.vis_fn = vis_fn
        self.batch_size = batch_size
        pipeline_list = augmenters
 
        if codec is not None :
            if isinstance(codec, dict) :
                #self.codec = KEYPOINT_CODECS.build(copy.deepcopy(codec))
                #codec = {**codec, **dict(codec_type='encode')}
                self.codec = CODECS.build(copy.deepcopy(codec))
            elif isinstance(codec, BaseCodec):
                self.codec =  codec
            else:
                raise TypeError(
                    f"codec_cfg must be 'dict' type, but got {type(codec)} @pose_dataloader "
                    "pose dataloader need to build completed codec including encoder and decoder"
                )  
            #self.batch_encode = lambda x :  self.codec(x,y_pred=None,codec_type='encoder')   
            assert isinstance(self.codec, Callable), "self.codec is not callable"      
        else:
            warnings.warn(
                "codec is None, you cannot gen targets used in traaining, but test_mode to review data is ok @pose_dataloader"
            ) 

        if hasattr(self,'codec'):
            assert isinstance(self.codec, Callable), \
            "self.codec.batch_encode is not callable"
            #pipeline_list = augmenters + [dict(type='EnsureTensor'), lambda x :  self.codec(x,y_pred=None,codec_type='encode')]
            pipeline_list = augmenters + [dict(type='EnsureTensor'), self.codec]
        else:
            pipeline_list = augmenters + [dict(type='EnsureTensor')] if ensure_to_tensor else  augmenters
        

        super().__init__(
                tfrec_datasets_list = tfrec_datasets_list,
                batch_size = batch_size,
                prefetch_size = prefetch_size,
                shuffle = shuffle,
                pipeline = pipeline_list,
                parallel_iterations = parallel_iterations
        )
        
    def get_pipeline_cfg(self, note : str = None) ->None:
        print(f"\n ----------------pipeline_cfg  (< ith > --- key : value)  {note if type(note)==str else ''}-------------------")
        for i,(key, value) in enumerate(self.pipeline.transforms_dict.items()):
            print(f"< {i+1} >  --- {key} : {value}") 
        print('--------------------------------------------------------------------------------\n')
      
    
    def get_num_TotalSamples(self, 
                            tfrecords_dir : str) -> int:
        return self.ds_TotalSamples 
    
    # def get_ds_TotalSamples_test(self) -> int:
    #     return self.tfrec_parser.get_TotalSamples(log=False)


    def encoder(
            self, data : dict
    ) -> dict:

        """ generate codec results (targets) 
            in order to meet your model's output shapes, pack imge, y_true and sample_weights
            for training
        Args:
            data (dict): data from preprocessing or argument function
        Returns:
            targets (tensor): generated ground true data for training
            i.e. : [ imgae, heatmap, sample_weights ]@ pose2d det
        """ 
        #if hasattr(self,"codec"): 
        assert hasattr(self,'codec'), '@dataloader.encoder'
        return self.codec(
            data, y_pred=None,codec_type='encode'
        )
    
    def decoder(
            self, data : Union[Tensor,List[Tensor],Dict], *args, **kwargs
    ) -> Union[Tensor,List[Tensor],Dict]:

        """ generate codec results (targets) 

        """ 
        assert hasattr(self,'codec'), '@dataloader.decoder'
        return self.codec(
            data, codec_type='decode', *args, **kwargs
        )  

    
    def DataAdapter(
            self ,                 
            data : Dict[str,Tensor],
            unpack_x_y_sample_weight: bool=False
    ) -> Union[Dict[str,Tensor],Tuple[Tensor]]:
        """ PackPoseInputs
        PackPoseInputs determine output format of tfds 

        Args:
            data (dict): data from preprocessing or argument function
            test_mode (bool): default to False
        Returns:
            output (dict) : if test_mode=True, 
                            for test_mode  ---Dict[str,Tensor]  is same as input format
            output (tuple): if test_mode=False, 
                            for training  ---Tuple[Tensor] to meet model.fit spec

        """
        if not unpack_x_y_sample_weight:
            return data
        
        resized_img = data['image']
        if data.get('y_true',None) is None :
            raise ValueError(
                  "data cannot find 'y_true' key"
            )
    
        y_true = data['y_true']
        if 'sample_weight' in data.keys():
            sample_weight = data['sample_weight']
            #Tuple[Tensor]
            return resized_img, y_true, sample_weight
        
        return resized_img, y_true
    

 
    def GenerateTargets(
            self, 
            test_mode=False, 
            unpack_x_y_sample_weight : bool = False,
            ds_weights : Optional[List[float]]=None,
            show_PackTensorTypeSpec : bool = False,
            show_pipeline_cfg  : bool = False
    )-> tf.data.Dataset:
        r""" get object of data generator
        arg:
           test_mode (bool) : for debug and metric test, 
                            if test_mode=True, dataloader cannot genertate enocoded targets for training
           ds_weights (List[float]) :

        Returns:
            dataset(object) : data generator 
        
        Note:    
            1. how to verify its content 
                for features in dataset.take(2):
                    features['kps']
            2. how to use it for training
               Model.fit(dataset)

        """   

        if  isinstance(ds_weights,list):
            self.ds_weights =  ds_weights
        
   
        #1 parse dataset and merge them if multi-dataset is aviable; 
        batch_dataset = self.prepare(
            parsers_list = self.parsers, 
            tfrec_RawData_list = self.tfrec_RawData_list, 
            meta_info = test_mode, 
            ds_weights = self.ds_weights
        )
        
        #2 pipeline : data augments/encode/PackPoseInputs pipeline
        self.pipeline.transforms_dict[self.DataAdapter.__name__] = lambda x : self.DataAdapter(x, unpack_x_y_sample_weight if hasattr(self,'codec') else False)
        batch_dataset = batch_dataset.map(self.pipeline, num_parallel_calls=tf.data.AUTOTUNE)

        #3 tfds ops : repeat, prefetch, cached
        if test_mode :
            'output dict[str,Tensor]'
            batch_dataset = batch_dataset.prefetch(self._prefetch)
            warnings.warn("test_mode cannot support training !!!!!!!!!! @pose_dataloader") 
        else:
            'output Tuple[Tensor]'
            batch_dataset = batch_dataset.repeat().prefetch(self._prefetch)
        
        if show_pipeline_cfg :
            self.get_pipeline_cfg() 
        if show_PackTensorTypeSpec:
            PackInputTensorTypeSpec(next(iter(batch_dataset)), {})
        
        return batch_dataset
    
    
    def explore_samples(
        self, take_batch_num : int =1, test_mode : bool = True, *arg,**kwargs
    ):
        if not isinstance(self.vis_fn, Callable):
            raise ValueError(
                f"vis_fn must be callable type @{self.__class__.__name__}, "
                f"but got {type(self.vis_fn)}"
            )
        
        batch_dataset = self.GenerateTargets(
            test_mode = test_mode, 
            unpack_x_y_sample_weight= False, 
            ds_weights =None
        )
        self.get_pipeline_cfg(note=' @explore_samples')
        
        self.vis_fn(
            [sample for sample in batch_dataset.take(take_batch_num)], *arg,**kwargs
        )
        del batch_dataset

    def execution_time(
        self, take_batch_num : Optional[int]=10
    ):
 
        from datetime import datetime
        batch_dataset = self.GenerateTargets(
            test_mode=False , 
            unpack_x_y_sample_weight= False, 
            ds_weights =None
        )
        total_execution_time = 0.
        tick = datetime.now()
        for i, _ in enumerate( batch_dataset.take(take_batch_num)):
            diff = datetime.now()  - tick 
            total_execution_time += diff.total_seconds()
            #print(f'execution time (s): {diff.total_seconds()} ---{i}')
        
        avg_execution_time = total_execution_time/take_batch_num
        print(f'avg execution time (s): {avg_execution_time}')
        del batch_dataset


