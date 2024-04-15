
from typing import Optional
import tensorflow as tf
from keras import backend as K
#-------------------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------------------
class EMA_ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    VERSION = '1.0.0'
    R""" ModelCheckpoint with ExponentialMovingAverage technique.
    date : 2024/1/26
    author : Dr. Shen 
    
    when you apply ExponentialMovingAverage technique,
    you should save two different weight files.
    one is navtive weights for training, other is ema weights for inference in general
    so, EMA_ModelCheckpoint is to save ema weights and ModelCheckpoint is to save native model weights

 
    Args:
        filepath (str) : path to save ema weigt
        decay (float) : A scalar float value. The decay parameter. Default to  0.999.
        num_updates(int) : Optional number of iterations to apply ema to update variables,  Default to None
        zero_debias (bool) : If True, zero debias moving-averages that are initialized with tensors.
            (Note: moving averages may not be initialized with non-variable tensors when eager execution is enabled)., 
            Defaults to False.
        val_use_ema_weights (bool) : whether to apply ema weights for validation dataset. Default to False,


    Note :
       - shadow_variable = decay * shadow_variable + (1 - decay) * variable
       - we need to save two files ; one is ema weights and other is trained_native_weights 
       - just only update ema weights by tf.train.ExponentialMovingAverage, don't overwrite them to mdoel in training progress
       -  ema_chekpoint_callback should be called before chekpoint_callback [ema_chekpoint, chekpoint]
          


    References:
        - (https://spaces.ac.cn/archives/6575/comment-page-1)
        - (https://stackoverflow.com/questions/72949031/transferring-exponential-moving-average-ema-of-tensorflow-custom-model-to-anot)
        - (https://github.com/tensorflow/models/issues/10452)
        - (https://gist.github.com/soheilb/c5bf0ba7197caa095acfcb69744df756)
        - (https://blog.csdn.net/qq_44817196/article/details/119817037)
        - (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/yolov8/losses.py#L329)
        -  more detail for tf.train.ExponentialMovingAverage :  https://github.com/tensorflow/tensorflow/blob/v2.14.0/tensorflow/python/training/moving_averages.py#L283-L689


    Example:
        '''Python
        ema_file_path = 'td-hm_HRNet-w32_udp-b64-200e_coco-256x192_EMA_Test.h5'
        file_path =  'td-hm_HRNet-w32_udp-b64-200e_coco-256x192_Test.h5'

        chekpoint = ModelCheckpoint(
            filepath = file_path,
            monitor="loss",
            verbose=1,
            save_best_only=False,
            save_weights_only=True
        )

        ema_chekpoint = ExponentialMovingAverage(
            ema_file_path,
            decay = 0.999, 
            val_use_ema_weights = True,
            verbose=1
        )   

        history = model.fit(
            batch_train_dataset, 
            epochs = 5, 
            steps_per_epoch = 4,
            validation_data = batch_val_dataset,
            validation_steps = 2,
            callbacks=[ema_chekpoint, chekpoint]
        )

    """
    def __init__(
        self, 
        filepath : str,
        decay : float = 0.999, 
        num_updates : Optional[int]=None,
        zero_debias : bool=False,
        val_use_ema_weights : bool = False,
        name = 'EMA_ModelCheckpoint',
        **kwargs
    ): 
        super().__init__(
            filepath, 
            save_weights_only=True, 
            save_freq='epoch', 
            **kwargs
        ) 
        self.decay = decay
        self.name = name
        # Create an ExponentialMovingAverage object
        self.ema = tf.train.ExponentialMovingAverage(
            decay=self.decay,
            num_updates = num_updates,
            zero_debias = zero_debias,
            name = self.name
        )
        self.val_use_ema_weights = val_use_ema_weights
        self.applying_ema_vars = False
        self.curr_native_vars = None

    def swap_to_ema_vars(self):
        self.applying_ema_vars = True
        self.curr_native_vars = K.batch_get_value(self.model.trainable_variables)
        averages = [self.ema.average(var) for var in  self.model.trainable_variables]
        K.batch_set_value(zip(self.model.trainable_variables, averages))

    def swap_to_native_vars(self):
        self.applying_ema_vars = False
        K.batch_set_value(zip(self.model.trainable_variables, self.curr_native_vars))
   
    def on_train_begin(self, logs=None):
        tf.print("apply ExponentialMovingAverage...............")
        'init ema before training'
        self.ema.apply(self.model.trainable_variables)

    # def on_train_end(self, logs=None):
    #     'init ema before training'
    #     self.native_vars = K.batch_get_value(self.model.trainable_variables) 

    # def on_batch_end(self, batch, logs=None):
    #     tf.print("hello!! --- (on_batch_end)")

    def on_train_batch_end(self, batch, logs=None):
        "update shadow variables in ema; doesn't modify trained weights"
        # if self.num_step_updates is not None and  batch % self.num_step_updates != 0:
        #     return
        #tf.print("batch", batch)
        self.ema.apply(self.model.trainable_variables) 
  
        
    def on_test_begin(self, logs=None):
        #tf.print("hello!! --- (on_test_begin)")
        if self.val_use_ema_weights == False:
            return
        'copy trained weights to temp'
        'load shadow variables to model'
        self.swap_to_ema_vars()
   

    # def on_test_end(self, logs=None):
    #     tf.print("hello!! --- (on_test_end)")

    def on_epoch_end(self,epoch, logs=None):
        if not self.applying_ema_vars:
            self.swap_to_ema_vars()
   
        super().on_epoch_end(epoch, logs)
        'swap to native weights for training in next epoch '
        if self.applying_ema_vars:
            self.swap_to_native_vars()
        # tf.print("hello!! --- (on_epoch_end)")
        # averages = [self.ema.average(var) for var in  self.model.trainable_variables]
        # tf.print(averages[0][0,0,0,:10])
            


