
from collections.abc import Callable 
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import tensorflow as tf
from tensorflow import Tensor
from lib.Registers import LOSSES

@LOSSES.register_module()
class PoseLifterRegLoss(tf.losses.Loss):

    VERSION = '1.0.0'
    r"""PoseLifterRegLoss


    """
    def __init__(self, 
                 w_mpjpe : Optional[List[float]] = [1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4],
                 root_id : int = 0,
                 use_sample_weight : bool = True,
                 **kwargs):
        super(PoseLifterRegLoss, self).__init__(reduction="auto", name="PoseLifterRegLoss")

        assert isinstance(root_id,int),"root_id must be int"
        assert isinstance(w_mpjpe,List), "w_mpjpe must be list"
        self.root_id = root_id
        if w_mpjpe:
            self.w_mpjpe = tf.constant(w_mpjpe, dtype=tf.float32)
        else:
            self.w_mpjpe = 1. 

        self.use_sample_weight = use_sample_weight

        super(PoseLifterRegLoss, self).__init__(**kwargs)

    def weighted_mpjpe(self, 
                    y_true : Tensor, 
                    y_pred : Tensor, 
                    sample_weight : Optional[Tensor]=None):
        
        norm_loss = tf.norm(y_pred-y_true,axis=-1) #(b,f,17)
        if self.use_sample_weight :
            weighted_norm_loss = norm_loss*self.w_mpjpe*sample_weight #(b,f,17)
            normalizer = tf.maximum(tf.math.count_nonzero(sample_weight, axis=[1,2],dtype=tf.float32), 1.)  #(b,f,17)=>(b)
            loss_3d_reg = tf.reduce_sum(weighted_norm_loss,axis=[1,2])/normalizer #(b,f,17)=>(b,) 
        else:
            weighted_norm_loss = norm_loss*self.w_mpjpe
            loss_3d_reg = tf.reduce_mean(weighted_norm_loss)
        return loss_3d_reg

    def __call__(self,                     
                y_true : Tensor, 
                y_pred : Tensor, 
                sample_weight : Optional[Tensor]=None): 
        """
        y_true : (b,f,17,3)
        y_pred : (b,1,17,3)
        sample_weight : (b,f,17)

        """
        y_true = tf.concat([y_true[:,:,:self.root_id,:],
                            tf.zeros_like(y_true[:,:,self.root_id:self.root_id+1,:]),
                            y_true[:,:,self.root_id+1:,:] ], axis=2) #(b,f,17,3)
        
        #y_true = tf.concat([ y_true[:,:,:8,:], tf.zeros_like(y_true[:,:,8:9,:]), y_true[:,:,9:,:] ], axis=2) #(b,f,17,3)
        loss = self.weighted_mpjpe(y_true, y_pred, sample_weight)

        return loss