
from collections.abc import Callable 
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import tensorflow as tf
from lib.Registers import LOSSES
############################################################################
#
#
############################################################################
@LOSSES.register_module()
class KeypointsMSELoss(tf.losses.Loss):
    VERSION = '1.0.0'
    r""" JointsMSELoss
    
    """
    def __init__(self, 
                joints_balance_weights : List[float], 
                loss_weight : float = 0.5,
                **kwargs):
        super().__init__(**kwargs)

        self.loss_weight = loss_weight
        self.mse = tf.losses.MeanSquaredError(reduction='none')
        '''
        'balence_weights '
        body_balance_weights = tf.constant([1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                                            1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5], dtype=tf.float32) 
        '''

        if joints_balance_weights == [] or joints_balance_weights==None :
           self.kps_balence_weights = 1.
        else:
           self.kps_balence_weights =  tf.constant(
               joints_balance_weights, dtype=tf.float32
            ) 

    def call(self, y_true, y_pred):
        y_pred_shape = tf.shape(y_pred)
        batch_size = y_pred_shape[0]
        num_of_joints = y_pred_shape[-1]

        y_pred = tf.reshape(
            tensor=y_pred, shape=[batch_size, -1, num_of_joints]
        )  
        y_true = tf.reshape(
            tensor=y_true, shape=[batch_size, -1, num_of_joints]
        ) #(b, h*w, n)
        y_pred = tf.transpose(y_pred,[0, 2, 1]) #(b, n, h*w)
        y_true = tf.transpose(y_true,[0, 2, 1]) #(b, n, h*w)
        loss = self.mse(y_true=y_true,y_pred=y_pred)*self.kps_balence_weights  #(batch,n)*(n) => (batch,n)
        return loss*self.loss_weight #(batch,n)
    


  
############################################################################
#
#
############################################################################
@LOSSES.register_module()
class MultiHeatmapMSELoss(tf.losses.Loss):
    VERSION = '1.0.0'
    r""" MultiHeatmapMSELoss
    
    """
    def __init__(self, 
        num_joints : int = 17,
        num_levels : int = 4,  
        use_ohkm : bool = False, 
        ohkm_top_k : int = 8,
        ohkm_level_index : int = 0,
        joints_balance_weights : Optional[List[float]] = None, 
        level_balance_weights :  Optional[List[float]] = None,    # [1.,1.,1.,1.]
        loss_weight : float = 1.,
        **kwargs):
        super().__init__(**kwargs)
        # loss : (batch, n, joints) => loss.mean() or  loss.sum()/(batch*n*joints)
        self.num_joints = num_joints
        self.num_levels = num_levels
        self.loss_weight = tf.cast(loss_weight , dtype=tf.float32) 

        'level_balance_weights'
        if level_balance_weights is not None :
            level_balance_weights = [float(num_levels) if (i==ohkm_level_index and use_ohkm) else 1. for i in range(num_levels)]
        else:
            if not isinstance(level_balance_weights, list) or len():
                raise TypeError(
                f"level_balance_weights must be 'list' type, but got {type(level_balance_weights)}"
                )
            if len(level_balance_weights)!=num_levels:
                raise ValueError(
                f"len(level_balance_weights) : {len(level_balance_weights)}, it must be equal to num_levels : {num_levels}"
                )

        self.level_balance_weights = tf.constant(
            level_balance_weights, dtype=tf.float32
        ) 
        'ohkm cfg (optional)'
        self.use_ohkm = use_ohkm
        if self.use_ohkm :
            if ohkm_top_k > num_joints:
                raise ValueError ("top_k must be less than num_joints")
            self.ohkm_level_index = (ohkm_level_index+self.num_levels ) if ohkm_level_index<0 else ohkm_level_index
            self.level_indices = [i for i in range(num_levels) if i!=self.ohkm_level_index]
            self.ohkm_top_k = ohkm_top_k
            self.ohkm_factor = tf.cast( 
                self.num_joints/self.ohkm_top_k, dtype=tf.float32
            ) #scalar
            
            self.multi_one_hot = tf.keras.layers.CategoryEncoding(
                num_tokens=num_joints, output_mode='multi_hot'
            ) #note: only support rank=2 if  output_mode='multi_hot'

        if joints_balance_weights is not None :
            self.kps_balence_weights =  tf.constant(
               joints_balance_weights, dtype=tf.float32
            )
        else:
            self.kps_balence_weights = 1.
        'loss fun'

        self.mse = tf.losses.MeanSquaredError(reduction='none') 

         
    def call(self, y_true, y_pred):
        
        y_pred_shape = tf.shape(y_pred)
        batch_size = y_pred_shape[0]
        #num_heatmaps = y_pred_shape[1]


        tf.debugging.assert_equal(
            y_pred_shape[-1], self.num_joints,
            message=f'input.shape[-1] must be equal to num_joints={self.num_joints}' 
        )
        tf.debugging.assert_equal(
            y_pred_shape[1], self.num_levels,
            message=f'input.shape[1] is num of feat., it must be equal to num_levels={self.num_levels}' 
        )   

        y_pred = tf.reshape(
            tensor=y_pred, shape=[batch_size, self.num_levels, -1,  self.num_joints]
        )  # (batch, 4, h*w, joints)
        y_true = tf.reshape(
            tensor=y_true, shape=[batch_size, self.num_levels, -1,  self.num_joints]
        )  # (batch, 4, h*w, joints)
        y_pred = tf.transpose(
            y_pred,[0, 1, 3, 2]
        ) #(b, 4, joints, h*w)
        y_true = tf.transpose(
            y_true,[0, 1, 3, 2]
        ) #(b, 4, joints, h*w)

        loss = 0.0
        if self.use_ohkm :
            # ohkm_pred, preds = tf.split(
            #     y_pred,[1,3],axis=1
            # ) 
            # ohkm_true, trues = tf.split(
            #     y_true,[1,3],axis=1
            # )
            preds = tf.gather(
                y_pred, self.level_indices, axis=1
            )#(b, 3, joints, h*w)
            ohkm_pred = y_pred[:,self.ohkm_level_index,...] #(b, joints, h*w)

            trues = tf.gather(
                y_true, self.level_indices, axis=1
            )#(b, 3, joints, h*w)
            ohkm_true = y_true[:,self.ohkm_level_index,...]  #(b, joints, h*w)


            loss = self.mse(
                y_true=trues, y_pred=preds
            ) #(batch, 3, joints)
            ohkm_loss = self.mse(
                y_true=ohkm_true, 
                y_pred=ohkm_pred,
            ) #(batch, joints)
            _ , topk_idx = tf.math.top_k(
                ohkm_loss, k=self.ohkm_top_k 
            ) #(batch, top_k)
            topk_idx = self.multi_one_hot(
                topk_idx 
            )#(batch, top_k) => (batch, joints) 
            ohkm_loss = tf.where(
                tf.equal(topk_idx, 1.), ohkm_loss, 0.
            ) #(batch, joints)
            ohkm_loss = tf.expand_dims(
                ohkm_loss, axis=1
            ) #(batch, 1, joints)
            # ohkm_factor = tf.cast( 
            #     self.num_joints/self.ohkm_top_k, dtype=tf.float32
            # ) #scalar
            loss = tf.concat(
                [ohkm_loss*self.ohkm_factor,loss], 
                axis=1
            ) #(batch, 4, joints)
        else:
            loss = self.mse(y_true=y_true,y_pred=y_pred)  ##(b, 4, joints, h*w)=>(b, 4, joints)
        #loss : (batch, 4, n)
        return loss*self.kps_balence_weights*self.level_balance_weights[:,None]*self.loss_weight  
