# Copyright (c) Movella  All rights reserved.
from collections.abc import Callable 
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from tensorflow import Tensor
import tensorflow as tf
from lib.utils.common import is_path_available
from lib.Registers import TRANSFORMS,DATASETS
import copy

from collections import OrderedDict
from inspect import isfunction
############################################################################
#
# 
############################################################################
class Compose:
    VERSION = '1.0.2'
    r"""Compose multiple transforms sequentially.
    buid up data pipeline by composing transfoms

    References:
        - [Based on the mmengine ( https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/base_dataset.py )]

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform objects or config dicts 
                   if items in transforms is objects, it msut be callable
                   if items in transforms is  config dicts , it will be built automatically

    Example:
        '''python
        dataset
        train_pipeline = [
                    dict(type='RandomFlip', direction='horizontal'),
                    dict(type='RandomHalfBody'),
                    dict(type='TopdownAffine', input_size=codec['input_size']),
                    ]
        tansform = Compose(train_pipeline)
        tansformed_dataset = tf.data.dataset.map(tansform)
        '''
    """

    def __init__(self, 
                transforms: Optional[List[Union[dict, Callable]]]=None,
                parallel_iterations : int = 16
        ):
        self.parallel_iterations = parallel_iterations

        self.transforms_dict = OrderedDict([])
        if transforms is None:
            transforms = []
      
        for transform in transforms:
            self.transforms_dict = self.add_transform(transform)
            
            
    def add_transform(self, 
                      transform):
        if isinstance(transform, dict):
            #transform['parallel_iterations'] = self.parallel_iterations
            transform = TRANSFORMS.build(transform)
            if hasattr(transform, 'parallel_iterations'):
                transform.parallel_iterations = self.parallel_iterations
            
            if not callable(transform):
                raise TypeError(f'transform should be a callable object, '
                                f'but got {type(transform)}')
            self.transforms_dict[transform.__class__.__name__] = transform

        elif callable(transform):
            if isfunction(transform):
                ''' function
                    def fun(data):
                '''
                self.transforms_dict[transform.__name__] = transform 
            else:
                ''' class
                  class trans:
                    def __init__(*arg,**kwargs):

                    def __call__(self,data):
                '''
                self.transforms_dict[transform.__class__.__name__] = transform 
        else:
            raise TypeError(
                    f'transform must be a callable object or dict, '
                    f'but got {type(transform)} @@Compose.add_transform') 
        
        return  self.transforms_dict 
    
    def remove_transform(self, 
                         transform):
        removed_key = None
        if isinstance(transform, dict):
            removed_key = transform['type']
        elif callable(transform):
            removed_key = transform.__name__
        elif isinstance(transform, str):
            removed_key = transform
        else:
            raise TypeError(
                    f'transform must be a callable object or dict, '
                    f'but got {type(transform)} @Compose.remove_transform') 
        
        if transform['type'] in self.transforms_dict.keys():
            _ = self.transforms_dict.pop(removed_key)
        else:
            raise ValueError(
                    f"no such key '{removed_key}' in cuurent transforms_dict @Compose.remove_transform") 

        return self.transforms_dict
        
        
    def __call__(self, data: Dict[str,Tensor]) -> Optional[Dict[str,Tensor]]:
        """Call function to apply transforms sequentially.
        Args:
            data (dict): A result dict contains the data to transform.
        Returns:
            dict: Transformed data
        """
        # for t in self.transforms_list:
        #     data = t(data)
        #     if data is None:
        #         return None
        for key ,t in self.transforms_dict.items():
            data = t(data)
            if data is None:
                return None
        return data
    
    def get_config(self):
        return self.transforms_dict.keys()
    

    



############################################################################
#
# 
############################################################################
class tfds_Base(metaclass=ABCMeta):
    version = '1.0.1'
    r""" tfds_Base , base features to generate tf.data.Dataset by loading TFRecords
    Only suport to load TFRecords
    1. support to load multi-datasets and automatically merge as one datasets with weights.
    2. support datasets with ragged batch to save memory
    3. build up data pipeline with transform functions to do data augment after batching



    TO DO:
        1. resize sample to the same size before batching
        2. shuffle_buffer
        3. reshuffle_each_iteration
        4. implement GenerateTargets as general example

    Args:
        tfrec_dataset (List[dict]):  parse specific dataset and get its total smaples
        batch_size (int):  how many instances to include in batches after loading.
                Should only be specified if img_size is specified (so that images
                can be resized to the same size before batching).
        prefetch_size (int): prefetch_size on batch size
        shuffle (bool) : whether to shuffle the dataset, defaults to True
        shuffle_buffer: the size of the buffer to use in shuffling.
        pipeline (list): data aguments used in dataset

    """

    def __init__(self,
            tfrec_datasets_list : List[Dict],
            batch_size : Optional[int]=256,
            prefetch_size : Optional[int]=4,
            shuffle  :  Optional[bool] = True,
            shuffle_buffer : Optional[bool] =None, 
            pipeline: List[Union[dict, Callable]] = [],
            parallel_iterations : int = 16,
            ): 
        
        'basic config in pipeline'
        self._batch = batch_size
        self._prefetch = prefetch_size
        self._shuffle = shuffle 
        if self._shuffle :
            self._shuffle_buffer = shuffle_buffer or 8 * batch_size

 
        'tfrec_dataset dir , use deepcopy to avoid missing items in cfg due to using pop() ' 

        'raw_tfrec_dataset(object)'
        'get parsers list for datasets'
        self.parsers = [ DATASETS.build(parser_cfg) for parser_cfg in tfrec_datasets_list ]
        'get tfrec_dir_list of  datasets'
        self.tfrec_dir_list =  [ parser.tfrecords_dir for parser in self.parsers]
        'get tfrec_datasetList ( List[tf.data.Dataset] ) '
        self.tfrec_RawData_list = [ self.load_tfrec(tfrecords_dir) for tfrecords_dir in self.tfrec_dir_list]


        'data argumenets and transforms'
        self.pipeline = Compose(pipeline, parallel_iterations)
        self.sample_keys = None

        'get_ds_TotalSamples'
        self.ds_Samples_list = [ parser.TotalSamples for parser in self.parsers] 
        self.ds_TotalSamples = sum(self.ds_Samples_list)
        self.ds_steps_per_epoch = self.ds_TotalSamples//self._batch #steps_per_epoch 
        self.ds_weights =   [float(ds_Samples/self.ds_TotalSamples) for ds_Samples in self.ds_Samples_list] 
        

  
    @abstractmethod
    def get_num_TotalSamples(self, tfrecords_dir : str):
        r""" get total smaple amounts of loded TFRecords
        Args:
            tfrec_dir (str): the folder's dir of TFRecords.
        Returns:
            total smaples (int): total smaples of all TFRecords 
        """
        return NotImplemented
    
    
    def load_tfrec(self, tfrec_dir : str) ->  object:
        r""" Loads the dataset from TFRecords 
        Args:
            tfrec_dir (str): the folder's dir  of TFRecords.
        Returns:
            raw_dataset (object): tf.data.Dataset  containing loaded TFRecords ;
        """
        is_path_available(tfrec_dir)
        'init'
        autotune = tf.data.AUTOTUNE 
        'load data form "tfrecords_dir"'
        filenames = tf.io.gfile.glob(f"{tfrec_dir}/*.tfrec")
        raw_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=autotune)
        raw_dataset = raw_dataset.apply(tf.data.experimental.ignore_errors())  #保平安, good luck
        #return tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE).ignore_errors()
        return raw_dataset
        



    def prepare(self, 
                parsers_list :List,
                tfrec_RawData_list : List[tf.data.Dataset], 
                meta_info : bool = False,
                ds_weights :List[float] = [1.] ) ->  tf.data.Dataset:
        
        r"""prepared batched datasets 
        Get parsed data for tfrec data and merge multi-datasets
   
        to add some common info. used in transform pipeline
        i.e. image_size, kps_vis,....
        Args:
            data (dict): parsed data ; dict(img, kps, bbox)
        Returns:
            dict (dict): prepared data ; dict(img, kps, bbox,img_shape,kps_vis)


        1. parse loaded tfrec rawdata 
        2. and if multi-datasets is available, merge these datasets as one tf.data.Dataset objects 

        Args:
            parsers_list (List): native tfrec_data need to parse its content
            tfrec_RawData_list (List[tf.data.Dataset]) : A non-empty list of tf.data.Dataset objects with compatible structure
            meta_info(bool) : flag to determine whether output meta information in data pipeline
            ds_weights : represents the probability to sample from tfrec_RawData_list[i]

        Returns:
            batch_dataset(tf.data.Dataset): parsed and merged tf.data.Dataset objects with tf.raggged_tensor

        """
        parsed_datasets_list = [ tfds_RawData.map(lambda x : parser(x, meta_info=meta_info),tf.data.AUTOTUNE) \
                            for parser,  tfds_RawData in zip(parsers_list, tfrec_RawData_list) ]

        assert len(ds_weights)==len(parsed_datasets_list), "len(ds_weights) must be len(parsed_datasets_list) @tfds_Base.parse_and_merge"
        merged_dataset = tf.data.Dataset.sample_from_datasets(parsed_datasets_list, weights=ds_weights)

        if self._shuffle:
            merged_dataset = merged_dataset.shuffle(self._batch*4)

        if tf.__version__ > '2.10.1':
            batch_dataset = merged_dataset.ragged_batch(batch_size=self._batch)
        else:
            batch_dataset = merged_dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=self._batch ))
    
        # if tf.__version__=='2.10.0':
        #     batch_dataset = merged_dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=self._batch ))
        # else:
        #     batch_dataset = merged_dataset.ragged_batch(batch_size=self._batch)

        return batch_dataset
    

    @abstractmethod
    def encoder(self, data : dict):
        """ generate codec results (targets) for training

        in order to meet your model's outputs' shapes, you might concat your tensors
        Args:
            data (dict): data from preprocessing or argument function
        Returns:
            targets (tensor): generated ground true data for training
            i.e. : [ imgae, heatmap, sample_weights ]@ pose2d det
            i.e. : [ imgae, (regression, classification), sample_weights] @ object det 
        """   
        return NotImplemented
    
 
    @abstractmethod
    def GenerateTargets(self, 
                    test_mode=False, 
                    ds_weights : Optional[List[float]]=None)-> tf.data.Dataset:

        return NotImplemented