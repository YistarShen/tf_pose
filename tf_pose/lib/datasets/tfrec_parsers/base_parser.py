from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable
import tensorflow as tf
from lib.utils.common import is_path_available
from tensorflow import Tensor


r""" 
BaseCocoStyle structure: type=dict str={
    'image': TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name=None), 
    'bbox': TensorSpec(shape=(4,), dtype=tf.float32, name=None), 
    'kps': TensorSpec(shape=(17, 3), dtype=tf.float32, name=None), 
    'image_size': TensorSpec(shape=(2,), dtype=tf.int32, name=None), 
    'bbox_format': TensorSpec(shape=(), dtype=tf.string, name=None), 
    'meta_info': {
        'src_image': TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name=None), 
        'src_height': TensorSpec(shape=(), dtype=tf.int64, name=None), 
        'src_width': TensorSpec(shape=(), dtype=tf.int64, name=None), 
        'src_keypoints': TensorSpec(shape=(None,), dtype=tf.float32, name=None), 
        'src_bbox': TensorSpec(shape=(4,), dtype=tf.float32, name=None), 
        'src_num_keypoints': TensorSpec(shape=(), dtype=tf.int64, name=None), 
        'area': TensorSpec(shape=(), dtype=tf.float32, name=None), 
        'category_id': TensorSpec(shape=(), dtype=tf.int64, name=None), 
        'id': TensorSpec(shape=(), dtype=tf.int64, name=None), 
        'image_id': TensorSpec(shape=(), dtype=tf.int64, name=None), 
        'iscrowd': TensorSpec(shape=(), dtype=tf.int64, name=None)
    }, 
    'transform2src': {'scale_xy': TensorSpec(shape=(2,), dtype=tf.float32, name=None), 
                  'pad_offset_xy': TensorSpec(shape=(2,), dtype=tf.float32, name=None), 
                  'bbox_lt_xy': TensorSpec(shape=(2,), dtype=tf.float32, name=None)
    }
}
"""
# -------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------
class BaseParser(metaclass=ABCMeta):
    VERSION = '1.0.2'
    r"""BaseParser to parse TFRecords
    All Parser inherit this this class
    store 


    Args:
        data_root (optinal[str]) :  the folder's dir  of TFRecords.

    

    """


    def __init__(
        self, data_root : Optional[str]=None
    ):
   
        self.tfrecords_dir = data_root
        is_path_available(self.tfrecords_dir)
        self.TotalSamples = self.get_TotalSamples(log=True)

    def get_TotalSamples(
            self, log :bool = False
        ) ->int:  
        """ get total smaple amounts of loded TFRecords
            arg :

            Returns:

        """
        import numpy as np
        filenames = tf.io.gfile.glob(
            f"{self.tfrecords_dir}/*.tfrec"
        )
        total_items = 0
        if log : print("\r\n ========================================")
        for tfrec_name in filenames:
            raw_dataset = tf.data.TFRecordDataset(
                tfrec_name, num_parallel_reads=tf.data.AUTOTUNE
            )
            samples_in_tfrec = raw_dataset.reduce(
                np.int64(0), lambda x, _: x + 1
            )
            total_items += samples_in_tfrec
            tfrec_name = tfrec_name.replace(
                self.tfrecords_dir+"/",''
            )
            if log : print(f"samples_in_tfrec : {samples_in_tfrec} @ <{tfrec_name}>")
        if log : print(f"total_samples : {total_items} @< {self.tfrecords_dir} >\n")
        return total_items
    
    def get_data_root(self):
        return self.tfrecords_dir
    
    def gen_tfds(
            self, batch_size : int = 0, use_ragged_batch: bool = False, meta_info : bool=True
    ) ->tf.data.Dataset:
        
        #assert self.test_mode ,"only support test mode to gen tf.data.Dataset @BaseCocoStyleDataset_Parser.gen_tfds"
        is_path_available(self.tfrecords_dir)
        assert isinstance(batch_size, int),"batch_size must be int type"

        'load data form "tfrecords_dir"'
        filenames = tf.io.gfile.glob(
            f"{self.tfrecords_dir}/*.tfrec"
        )
        raw_dataset = tf.data.TFRecordDataset(
            filenames, num_parallel_reads=tf.data.AUTOTUNE
        )
        raw_dataset = raw_dataset.apply(
            tf.data.experimental.ignore_errors()
        ) 
        parsed_dataset = raw_dataset.map(
            lambda x : self.__call__(x, meta_info=meta_info), 
            num_parallel_calls=tf.data.AUTOTUNE
        )

        'set batch size for dataset'
        if batch_size:
            if use_ragged_batch :
                if tf.__version__=='2.10.0':
                    parsed_batch_dataset = parsed_dataset.apply(
                        tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size )
                    )
                else:
                    parsed_batch_dataset = parsed_dataset.ragged_batch(batch_size=batch_size)
            else:
                parsed_batch_dataset =  parsed_dataset.padded_batch(batch_size=batch_size)
            return parsed_batch_dataset.prefetch(batch_size*2)
        return parsed_dataset

    def explore_samples(
        self, take_batch_num : int =1, test_mode : bool = True, *arg,**kwargs
    ):
        from lib.visualization import Vis_SampleTransform
     
        batch_dataset = self.gen_tfds(
            batch_size = kwargs.pop('batch_size',1),
            use_ragged_batch = kwargs.pop('use_ragged_batch',True), 
            meta_info = test_mode
        )
        # batch_dataset = self.gen_tfds(
        #     batch_size = batch_size,
        #     use_ragged_batch = use_ragged_batch, 
        #     meta_info = meta_info
        # )

        Vis_SampleTransform(figsize=(20,10))(
           batched_samples_list = [sample for sample in batch_dataset.take(take_batch_num)],  *arg,**kwargs
        )
        # Vis_PoseSampleTransform(
        #     [sample for sample in batch_dataset.take(take_batch_num)],  *arg,**kwargs
        # )
        del batch_dataset

    @abstractmethod
    def __call__(
        slef, example : dict,meta_info: bool=False
    )->Tensor:
        return NotImplemented
    

