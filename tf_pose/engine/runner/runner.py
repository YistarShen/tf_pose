import os
from collections.abc import Callable 
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from lib.models.builder import build_pose_estimator
from lib.Registers import DATASETS

import os
from collections.abc import Callable 
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import tensorflow as tf
from lib.models.builder import build_pose_estimator
from lib.Registers import DATASETS


import warnings
import copy

class Runner:
    r""" A training helper for PyTorch.
        train_cfg = dict(epochs = 100,
                        param_scheduler = function
                        callbacks_cfg =  [dict( type = 'ModelCheckpoint;,
                                                filepath="str",
                                                monitor="loss",
                                                save_best_only = True,
                                                save_weights_only = True,
                                                verbose = 1)]

    """
    def __init__(self,
            model_cfg: Union[tf.keras.Model, Dict],
            main_dir: Optional[str]=None,
            train_dataloader: Optional[Union[tf.data.Dataset, Dict]] = None,
            val_dataloader: Optional[Union[tf.data.Dataset, Dict]] = None,
            train_cfg: Optional[Dict] = None):
        '''
        if isinstance(train_dataloader, tf.data.Dataset):
           raise RuntimeError("not implement : type of train_dataloader is tf.data.Dataset ")  
        
        '''


        '1 verfiy arg type'
        assert isinstance(train_dataloader, (tf.data.Dataset,dict) ) or train_dataloader is None, \
        f"vaild type of  train_dataloader must be tf.data.Dataset | dict type | None , but it's { type(train_dataloader)} @Runner"
 
        assert isinstance(val_dataloader, (tf.data.Dataset,dict) ) or val_dataloader is None, \
        f"vaild type of  val_dataloader must be tf.data.Dataset | dict type | None , but it's { type(val_dataloader)} @Runner"
         
        '#2  :  laod compiled tf model'
        #self.model = build_pose_estimator(model_cfg).get_model()
        self.model = build_pose_estimator(model_cfg)
        '#3  :  Train_Dataloader'        
        self.train_dataset = None
        self.train_steps_per_epoch = None

        if isinstance(train_dataloader, dict):
            self.train_dataset, self.tfds_train_builder, self.train_steps_per_epoch = self.build_tfds_from_cfg(train_dataloader)  
            batch_szie = self.tfds_train_builder._batch
        elif isinstance(train_dataloader, tf.data.Dataset):
            self.train_dataset = train_dataloader 
            self.train_steps_per_epoch = train_cfg['train_steps_per_epoch']
            batch_szie = next(train_dataloader.__iter__())[0].shape[0]
        else:
            batch_szie = None
            warnings.warn("train_dataloader and  batch_szie is None!!!!!!!!, you need to set up train_dataloader  @Runner")
    
        
        '#4  :  val_Dataloader'
        self.val_dataset = None
        self.val_steps_per_epoch = None 
        if isinstance(val_dataloader, dict):
           self.val_dataset, self.tfds_val_builder, self.val_steps_per_epoch = self.build_tfds_from_cfg(val_dataloader)
           #val_batch_szie = self.tfds_val_builder._batch
        elif isinstance(val_dataloader, tf.data.Dataset) :
           self.val_dataset = val_dataloader
           self.val_steps_per_epoch  = train_cfg['val_steps_per_epoch']
           #val_batch_szie = next(val_dataloader.__iter__())[0].shape[0]
        else:
            warnings.warn("val_dataloader is None!!!!, => val_dataset=val_steps_per_epoch=None @Runner")

   
        '#5  :  basic config to meet requirements for training'
        self.batch_szie = batch_szie 
        self.epochs = train_cfg["epochs"] 
        self.base_lr = self.model.optimizer.learning_rate

        if isinstance(train_cfg['callbacks_cfg'],List) :
            self.callbacks_list  = self.set_tf_callbacks(copy.deepcopy(train_cfg['callbacks_cfg']) )
        else:
            self.callbacks_list  = None

            
    def batch_decode(self, preds : tf.Tensor) ->  tf.Tensor:
        if hasattr(self, 'tfds_train_builder') and self.tfds_train_builder is not None:
            return self.tfds_train_builder.codec.batch_decode(preds)
        else:
            raise RuntimeError("batch_decode no implement @Runner.batch_decode") 
    
    def tfds_builder(self):
        if hasattr(self, 'tfds_train_builder') and self.tfds_train_builder is not None:
            return self.tfds_train_builder
        else:
            raise RuntimeError("no tfds_train_builder @Runner.tfds_builder") 
        

    def set_tf_callbacks(self, callbacks_cfg:List[Dict])->List[object]:
        print(f'\nset tf callbacks functions.................')
        callback_list = []
        if len(callbacks_cfg)!= 0:
            for callback_cfg in callbacks_cfg:
                print(f'already set < tf.keras.callbacks.{callback_cfg["type"]} >')
                '''
                if callback_cfg['type']== 'ModelCheckpoint' and self.saved_path is not None :
                   callback_cfg['filepath'] = self.saved_path
                '''
                callback_list.append( getattr(tf.keras.callbacks, callback_cfg.pop('type')) (**callback_cfg) )
            return callback_list
        else:
            print(f'No callback functions')
            return None
        
    def build_tfds_from_cfg(self,
                        dataloader: Dict,
                        test_mode : bool = False)->Tuple[tf.data.Dataset,int,int]:
        
        assert isinstance(dataloader, Dict) ,"dataloader must be dict type @build_tfds_from_cfg"
        
        
        tfds_builder  = DATASETS.build(copy.deepcopy(dataloader))
        ds_TotalSamples = tfds_builder.ds_TotalSamples
        steps_per_epoch = ds_TotalSamples//(tfds_builder._batch) 
        dataset = tfds_builder.GenerateTargets(test_mode=test_mode)  

        return  dataset, tfds_builder, steps_per_epoch

    def set_train_epochs(self, epochs:int):
        self.epochs = epochs
               
    def set_train_steps_per_epoch(self, 
                                train_steps_per_epoch : int) ->int : 
        self.train_steps_per_epoch = train_steps_per_epoch
    
    def set_val_steps_per_epoch(self, 
                                val_steps_per_epoch : int) ->int : 
        self.val_steps_per_epoch = val_steps_per_epoch 
    
    def get_batch_size(self):
        return self.batch_szie
    '''
    def cfg(self):
    '''

    def train(self, 
            epochs : Optional[int] = None,
            train_dataset: Optional[tf.data.Dataset] =  None,
            val_dataset: Optional[tf.data.Dataset] =  None,
            train_steps_per_epoch : Optional[int] = None,
            val_steps_per_epoch : Optional[int] = None,
            callbacks_list :Optional[List] = None):

        """training for TENSORFLOW.
        >>>    model.fit(
        >>>            x=None,
        >>>            y=None,
        >>>            batch_size=None,
        >>>            epochs=1,
        >>>            verbose='auto',
        >>>            callbacks=None,
        >>>            validation_split=0.0,
        >>>            validation_data=None,
        >>>            shuffle=True,
        >>>            class_weight=None,
        >>>            sample_weight=None,
        >>>            initial_epoch=0,
        >>>            steps_per_epoch=None,
        >>>            validation_steps=None,
        >>>            validation_batch_size=None,
        >>>            validation_freq=1,
        >>>            max_queue_size=10,
        >>>            workers=1,
        >>>            use_multiprocessing=False
        >>>    )
        """

        if isinstance(epochs, int): 
            self.epochs =  epochs

        if isinstance(train_dataset, tf.data.Dataset): 
            self.train_dataset =  train_dataset 

        if isinstance(train_steps_per_epoch, int) and isinstance(self.train_dataset, tf.data.Dataset): 
            self.train_steps_per_epoch =  train_steps_per_epoch 

        if isinstance(val_dataset, tf.data.Dataset): 
            self.val_dataset =  val_dataset 

        if isinstance(val_steps_per_epoch, int) and isinstance(self.val_dataset, tf.data.Dataset): 
            self.val_steps_per_epoch =  val_steps_per_epoch 
    
        if (isinstance(callbacks_list, list) and len(callbacks_list)!= 0 ) :
            callbacks_list = callbacks_list 
        else :
            callbacks_list = self.callbacks_list 
      
        'verify train_dataset/train_steps_per_epoch both are not None type and meet type spec'

        assert isinstance(self.epochs, int), \
        f'epochs must be int type, but it is {type(self.epochs)} @runner.train'
        assert isinstance(self.train_dataset, tf.data.Dataset), \
        f'type of train_dataset must be tf.data.Datase, but it is {type(self.train_dataset)} @runner.train' 
        assert isinstance(self.train_steps_per_epoch, int), \
        f'train_steps_per_epoch must be int type, but it is {type(self.train_steps_per_epoch)} @runner.train' 


        self.model.fit(self.train_dataset,
                       epochs = self.epochs,
                       steps_per_epoch = self.train_steps_per_epoch,
                       validation_data = self.val_dataset,
                       validation_steps = self.val_steps_per_epoch,    
                       callbacks = self.callbacks_list
                    )       
        
    def test(self):
        return NotImplemented
    
    @tf.function
    def __call__(self, tensor : tf.Tensor)->Any:
        '''
        
        NEED TO DO MERGE PREPROCESS AND Postptocess
        '''
        return self.model(tensor)
    
    def eval(self,
            eval_dataloader : dict, 
            batch_size : int = 32):
        '''
        now only support pose detection for base coco dataset

        EXAMPLE :
            eval_dataloader_cfg =  dict(
                type = 'pose_dataloader',
                batch_size = 32,
                prefetch_size = 4,
                shuffle  =  False,
                tfrec_dataset = dict(
                    type = 'BaseCocoStyleDataset_Parser', 
                    data_root = val_tfrec_kps_dir,
                ),
                pipeline = [affine_cfg],
                codec = None
            )

        '''
        if isinstance(eval_dataloader, dict):
            eval_dataset, tfds_eval_builder, eval_steps_per_epoch = self.build_tfds_from_cfg(eval_dataloader, test_mode=True)
            '''
            test_mode generate non-batched dataset, dataset should be tuple(transformed_gt, src_gt)
            '''
            eval_batched_dataset = eval_dataset.batch(batch_size).prefetch(batch_size)

        else:
            raise RuntimeError("eval_dataloader must be dict type @Runner.eval") 
        

        for idx, (transformed_gt, src_gt) in enumerate(eval_batched_dataset):
            print(idx)

            img = transformed_gt['image']

            hm_preds  = self.__call__(img)

            decoded_kps = tfds_eval_builder.codec.batch_decode(hm_preds)

            """
            go to coco metric
            """
        

        

      
    
    


        

