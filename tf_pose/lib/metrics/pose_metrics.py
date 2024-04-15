from collections.abc import Callable 
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import tensorflow as tf
from lib.Registers import METRICS


#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
@METRICS.register_module()
class PCKMetric(tf.keras.metrics.Metric):
    VERSION = "1.0.0"
    r"""PCKMetric.

    PCKMetric is used in training progress to monitor cuurent accuracy of pose estimation model
  
    Args:
        tf_dists (float): 
        num_kps (int): kernels used in each Conv2DTranspose if len(List) > 0
        hm_thr (List[float] | tf.tensor):  upsample ratio used in UpSampling2D if upsample_scale>0


    Example for coco_17kps:
        - pose_metric_cfg = dict(tf_dists = 0.5, 
                                num_kps = 17,
                                hm_thr= [0.75, 0.75, 0.75, 0.75, 0.75,
                                        0.25, 0.25, 0.25, 0.25, 0.25,
                                        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
                       
    """


    def __init__(self, 
                tf_dists : float=0.5, 
                num_kps : int=17, 
                hm_thr : Optional[ Union[List[float], tf.Tensor] ]= None, 
                prim_feat_index : int = 0,
                name='pck_acc', 
                **kwargs):
        
        super(PCKMetric, self).__init__(name=name, **kwargs)
        self.total_acc = self.add_weight(name='total_acc', initializer='zeros')
        self.total_sample = self.add_weight(name='total_avg_acc', initializer='zeros')

        self.thr = tf_dists
        self.num_kps = tf.cast(num_kps, dtype=tf.float32)
        self.acc = tf.zeros(shape=(num_kps,),dtype=tf.float32)
        self.avg_acc = tf.constant(0,dtype=tf.float32)

        'kps confidence threhold config'
        '''
        face_kps_thr = tf.ones(shape=(5,), dtype=tf.float32)*0.75
        body_kps_thr = tf.ones(shape=(12,), dtype=tf.float32)*0.25
        #feet_hands_kps_thr = tf.ones(shape=(10,), dtype=tf.float32)*0.25
        #self.hm_thr = tf.concat([face_kps_thr, body_kps_thr, feet_hands_kps_thr], axis=-1)
        self.hm_thr = tf.concat([face_kps_thr, body_kps_thr], axis=-1)

        tf.ones(shape=(17,), dtype=tf.float32)*0.5,
        '''

        'set hm_thr' 
        if isinstance(hm_thr, list) :
           self.hm_thr = tf.constant(hm_thr, dtype=tf.float32)
        elif isinstance(hm_thr, tf.Tensor) :
            self.hm_thr = hm_thr
        else:
            self.hm_thr = tf.ones(shape=(17,), dtype=tf.float32)*0.5,

        self.primary_feature_index = prim_feat_index

    def _nms(self, heatmap, kernel=5):
        hmax = tf.nn.max_pool2d(heatmap, kernel, 1, padding='SAME')
        keep = tf.cast(tf.equal(heatmap, hmax), tf.float32)
        return heatmap*keep

    def _kps_from_heatmap(self, batch_heatmaps, normalize=False):

        batch = tf.shape(batch_heatmaps)[0]
        height = tf.shape(batch_heatmaps)[1]
        width = tf.shape(batch_heatmaps)[2]
        n_points = tf.shape(batch_heatmaps)[3]

        batch_heatmaps = self._nms(batch_heatmaps) #(1,96,96,14)
        flat_tensor = tf.reshape(batch_heatmaps, (batch, -1, n_points)) #(1,96*96,14)

        # Argmax of the flat tensor
        argmax = tf.argmax(flat_tensor, axis=1) #(1,14)
        argmax = tf.cast(argmax, tf.int32) #(1,14)
        scores = tf.math.reduce_max(flat_tensor, axis=1) #(1,14)
        # Convert indexes into 2D coordinates
        argmax_y = tf.cast( (argmax//width), tf.float32)  #(1,14)
        argmax_x = tf.cast( (argmax%width), tf.float32)   #(1,14)
        if normalize:
            argmax_x = argmax_x / tf.cast(width, tf.float32)
            argmax_y = argmax_y / tf.cast(height, tf.float32)

        # Shape: batch * 3 * n_points  (1,14,3)
        batch_keypoints = tf.stack((argmax_x, argmax_y, scores), axis=2)
        'mask'
        mask =  tf.greater(batch_keypoints[:, :, 2], self.hm_thr) #(1,14)
        batch_keypoints = tf.where(tf.expand_dims(mask,axis=-1), batch_keypoints[:, :, :2], 0)
        batch_keypoints = tf.stack((batch_keypoints[...,0], batch_keypoints[...,1], scores), axis=2)

        return batch_keypoints

    def tf_calc_dists(self,  batch_kps_true, batch_kps_pred, normalize):

        cond = tf.greater(batch_kps_true[...,:2],(1., 1.)) #(batch,17, 2)
        mask = tf.reduce_all(cond, axis=-1)  #(batch,17)
        diff = batch_kps_pred - batch_kps_true #(batch,17, 2)
        dists = tf.where(mask, tf.math.reduce_euclidean_norm(diff/normalize, axis=-1) , -1.) #(batch,17)
        return dists

    def update_state(self, y_true, y_pred, sample_weight=None):
        """ 
        y_true : (b,levels, h,w, joints) / (b,h,w, joints)
        y_pred : (b,levels, h,w, joints) / (b,h,w, joints)
        """

        'support multi_heatmaps'
        if tf.rank(y_pred)==5:
            y_pred = y_pred[:, self.primary_feature_index]

        if tf.rank(y_true)==5:
            y_true = y_true[:, self.primary_feature_index]

        # y_pred = tf.where(
        #     tf.equal(tf.rank(y_pred),5), 
        #     y_pred[:, self.primary_feature_index], 
        #     y_pred
        # )
        # y_true = tf.where(
        #     tf.equal(tf.rank(y_true),5), 
        #     y_true[:, self.primary_feature_index],
        #     y_true
        # )
    
        batch_kps_pred = self._kps_from_heatmap(
            y_pred, normalize=False
        ) #(batch,17, 2)
        batch_kps_true = self._kps_from_heatmap(
            y_true, normalize=False
        ) #(batch,17, 2)

        tf_norm = tf.shape(y_true)[1:3]
        tf_norm = tf.cast(
            tf.reverse(tf_norm,axis=[0])/10, dtype=tf.float32
        ) #(2,)

        tf_dists = self.tf_calc_dists(
            batch_kps_pred[...,:2], batch_kps_true[...,:2], tf_norm
        ) #(batch,17)

        'effective samples per class'
        eff_samples_cond =  tf.not_equal(tf_dists, -1.)                            #(batch,17)
        eff_samples_per_class =  tf.reduce_sum( 
            tf.cast(eff_samples_cond, dtype=tf.float32), axis=0
        )   #(17,)

        'mask for accurate samples '
        accurate_samples_mask = tf.less(tf_dists, self.thr) & eff_samples_cond  #(batch,17)
        accurate_samples_per_cls = tf.reduce_sum(
            tf.cast(accurate_samples_mask, dtype=tf.float32), axis=0
        )     #(17,)

        'accuracy per calss'
        accuracy_per_cls= tf.where(
            tf.greater(eff_samples_per_class,0), accurate_samples_per_cls/eff_samples_per_class, 0.
        ) #(17,)
        'effective class'
        eff_cls_mask = tf.not_equal(accuracy_per_cls, 0.) #(17,)
        eff_cls_num = tf.reduce_sum( 
            tf.cast(eff_cls_mask, dtype=tf.float32), axis=0
        ) #(1,)

        "average accuracy for all effictive class per batch"
        #avg_accuracy = tf.reduce_sum(accuracy_per_cls, axis=0)
        if(eff_cls_num>0):
            avg_accuracy = tf.reduce_sum(accuracy_per_cls, axis=0)/(eff_cls_num + 1e-5)
        else:
            avg_accuracy = 0.

        self.avg_acc = avg_accuracy
        #print(self.avg_acc)

    def result(self):
        return self.avg_acc
    

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
@METRICS.register_module()
class SimCC_PCKMetric(tf.keras.metrics.Metric):
    r"""SimCC_PCKMetric.

    SimCC_PCKMetric is used in training progress to monitor cuurent accuracy of pose estimation model
  
    Args:
        tf_dists (float): 
        simcc_xy_dims (tuple(int)): to split features , (b,17,896)=> (b,17,512) and (b,17,384)
        num_kps (int): kernels used in each Conv2DTranspose if len(List) > 0
        hm_thr (List[float] | tf.tensor):  threshold value of heatmap to get effective kps


    Example for coco_17kps:
        - simcc_pose_metric_cfg = dict(tf_dists = 0.5, 
                                    simcc_xy_dims = (384, 512),
                                    num_kps = 17,
                                    hm_thr= tf.ones(shape=(17,), dtype=tf.float32)*0.25
                       
    """

    def __init__(self, 
            tf_dists : float = 0.5, 
            input_size_hw : tuple =(256,192), 
            num_kps : int = 17, 
            hm_thr : Optional[ Union[List[float], tf.Tensor] ]= None, 
            name='simcc_pck_acc', **kwargs):
        super(SimCC_PCKMetric, self).__init__(name=name, **kwargs)
        self.total_acc = self.add_weight(name='total_acc', initializer='zeros')
        self.total_sample = self.add_weight(name='total_avg_acc', initializer='zeros')

        self.thr = tf_dists
        self.num_kps = tf.cast(num_kps, dtype=tf.float32) 
        self.acc = tf.zeros(shape=(num_kps,),dtype=tf.float32) 
        self.avg_acc = tf.constant(0,dtype=tf.float32)
        self.size_splits = (input_size_hw[1]//4,input_size_hw[0]//4)
        'kps confidence threhold config'
        '''
        face_kps_thr = tf.ones(shape=(5,), dtype=tf.float32)*0.25
        body_kps_thr = tf.ones(shape=(12,), dtype=tf.float32)*0.25
        feet_hands_kps_thr = tf.ones(shape=(10,), dtype=tf.float32)*0.25
        #self.hm_thr = tf.concat([face_kps_thr, body_kps_thr, feet_hands_kps_thr], axis=-1) 
        self.hm_thr = tf.concat([face_kps_thr, body_kps_thr], axis=-1) 
        '''
        'set hm_thr' 
        if isinstance(hm_thr, list) and len(hm_thr)==num_kps:
           self.hm_thr = tf.constant(hm_thr, dtype=tf.float32)
        elif isinstance(hm_thr, tf.Tensor) and hm_thr.shape[0]==num_kps:
            self.hm_thr = hm_thr
        else:
            self.hm_thr = tf.ones(shape=(num_kps,), dtype=tf.float32)*0.5,
  
    def _nms(self, 
             heatmap, 
             kernel=5):
        hmax = tf.nn.max_pool2d(heatmap, kernel, 1, padding='SAME')
        keep = tf.cast(tf.equal(heatmap, hmax), tf.float32)
        return heatmap*keep
        
    def _kps_from_heatmap(self, 
                          batch_heatmaps, 
                          normalize=False):

        batch = tf.shape(batch_heatmaps)[0]
        height = tf.shape(batch_heatmaps)[1]
        width = tf.shape(batch_heatmaps)[2]
        n_points = tf.shape(batch_heatmaps)[3]

        batch_heatmaps = self._nms(batch_heatmaps) #(1,96,96,14)
        flat_tensor = tf.reshape(batch_heatmaps, (batch, -1, n_points)) #(1,96*96,14)

        # Argmax of the flat tensor
        argmax = tf.argmax(flat_tensor, axis=1) #(1,14)
        argmax = tf.cast(argmax, tf.int32) #(1,14)
        scores = tf.math.reduce_max(flat_tensor, axis=1) #(1,14)
        # Convert indexes into 2D coordinates
        argmax_y = tf.cast( (argmax//width), tf.float32)  #(1,14)
        argmax_x = tf.cast( (argmax%width), tf.float32)   #(1,14)
        if normalize:
            argmax_x = argmax_x / tf.cast(width, tf.float32)
            argmax_y = argmax_y / tf.cast(height, tf.float32)

        # Shape: batch * 3 * n_points  (1,14,3)
        batch_keypoints = tf.stack((argmax_x, argmax_y, scores), axis=2)
        'mask'
        mask =  tf.greater(batch_keypoints[:, :, 2], self.hm_thr) #(1,14)
        batch_keypoints = tf.where(tf.expand_dims(mask,axis=-1), batch_keypoints[:, :, :2], 0)
        batch_keypoints = tf.stack((batch_keypoints[...,0], batch_keypoints[...,1], scores), axis=2)

        return batch_keypoints

    def tf_calc_dists(self,  
                    batch_kps_true,
                    batch_kps_pred, 
                    normalize):

        cond = tf.greater(batch_kps_true[...,:2],(1., 1.)) #(batch,17, 2)
        mask = tf.reduce_all(cond, axis=-1)  #(batch,17)
        diff = batch_kps_pred - batch_kps_true #(batch,17, 2)
        dists = tf.where(mask, tf.math.reduce_euclidean_norm(diff/normalize, axis=-1) , -1.) #(batch,17)
        return dists

    def update_state(self, 
                    y_true : tf.Tensor, 
                    y_pred : tf.Tensor,
                    sample_weight = None):

        y_true = tf.transpose(y_true,perm=[0,2,1]) #(b,17,W+H)=>(b,W+H,17)
        y_pred = tf.transpose(y_pred,perm=[0,2,1])

        y_true = tf.nn.avg_pool1d(y_true,ksize=8,strides=8, padding='SAME') # (b,W+H,17) => (b,w+h,17)
        y_pred = tf.nn.avg_pool1d(y_pred,ksize=8,strides=8, padding='SAME') # (b,W+H,17) => (b,w+h,17)

        coord_pred_x, coord_pred_y = tf.split(y_pred, self.size_splits, axis=1) #(b,w+h,17) => (b,w,17) and (b,h,17)
        coord_true_x, coord_true_y = tf.split(y_true, self.size_splits, axis=1) #

        simcc_hm_yx_true = coord_pred_x[:,None,:,:]*coord_pred_y[:,:,None,:] #(b,h,w,17)
        simcc_hm_yx_pred = coord_true_x[:,None,:,:]*coord_true_y[:,:,None,:] #(b,h,w,17)

        batch_kps_pred = self._kps_from_heatmap(simcc_hm_yx_pred, normalize=False) #(batch,17,2)
        batch_kps_true = self._kps_from_heatmap(simcc_hm_yx_true, normalize=False) #(batch,17,2)

        tf_norm = tf.shape(simcc_hm_yx_true)[1:3] #(b,H,W)
        tf_norm = tf.cast( tf.reverse(tf_norm,axis=[0])/10, dtype=tf.float32) #(2,)

        tf_dists = self.tf_calc_dists(batch_kps_pred[...,:2], batch_kps_true[...,:2], tf_norm) #(batch,17)

        'effective samples per class'
        eff_samples_cond =  tf.not_equal(tf_dists, -1.)                            #(batch,17)
        eff_samples_per_class =  tf.reduce_sum( tf.cast(eff_samples_cond, dtype=tf.float32), axis=0)   #(17,)
    
        'mask for accurate samples '
        accurate_samples_mask = tf.less(tf_dists, self.thr) & eff_samples_cond  #(batch,17)
        accurate_samples_per_cls = tf.reduce_sum(tf.cast(accurate_samples_mask, dtype=tf.float32), axis=0)     #(17,)
    
        'accuracy per calss'
        accuracy_per_cls= tf.where(tf.greater(eff_samples_per_class,0), accurate_samples_per_cls/eff_samples_per_class, 0.) #(17,) 
        'effective class'
        eff_cls_mask = tf.not_equal(accuracy_per_cls, 0.) #(17,)  
        eff_cls_num = tf.reduce_sum( tf.cast(eff_cls_mask, dtype=tf.float32), axis=0) #(1,) 

        "average accuracy for all effictive class per batch"
        #avg_accuracy = tf.reduce_sum(accuracy_per_cls, axis=0) 
        if(eff_cls_num>0):
            avg_accuracy = tf.reduce_sum(accuracy_per_cls, axis=0)/(eff_cls_num + 1e-5)
        else:
            avg_accuracy = 0.

        self.avg_acc = avg_accuracy
        #print(self.avg_acc)

    def result(self):
        return self.avg_acc    