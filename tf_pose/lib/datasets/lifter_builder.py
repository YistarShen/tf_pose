from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import copy
import tensorflow as tf
from tensorflow import Tensor
import numpy as np



class tf_ChunkedGenerator():
  version = '2.0.0'
  def __init__(self,fetch_dataset,
            chunk_length, 
            joints_mask_prob = 0.5,
            upper_body_prob =0.5):
    print("chunk_length(GT):",chunk_length)

    cameras, poses_3d, poses_2d = fetch_dataset
    pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
    self.total_chunks = 0
    self.total_videos = len(poses_2d)
    for i, pose_2d in enumerate(poses_2d):
      n_chunks = (pose_2d.shape[0] + chunk_length - 1) // chunk_length
      offset = (n_chunks * chunk_length - pose_2d.shape[0]) // 2
      bounds = np.arange(n_chunks+1)*chunk_length - offset
      augment_vector = np.full(len(bounds - 1), False, dtype=bool)
      pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
      self.total_chunks += n_chunks

    'build tensor list'
    self.pairs = pairs  
    self.poses_2d_list = [ tf.convert_to_tensor(pose_2d, dtype=tf.float32) for pose_2d in poses_2d ]
    self.poses_3d_list = [ tf.convert_to_tensor(pose_3d, dtype=tf.float32) for pose_3d in poses_3d ]
    self.cameras = [tf.convert_to_tensor(cam, dtype=tf.float32) for cam in cameras ]
    assert len(self.poses_2d_list)==len(self.poses_3d_list), "'counts mismatch for poses_2d and poses_3d"

    'arguments'
    self.joints_mask_prob = joints_mask_prob
    self.upper_body_prob = upper_body_prob
    self.gt = chunk_length
    self.gt_ids = tf.range(start=0,limit=self.gt, delta=1, dtype=tf.int32)
    self.kps_num = 17
    self.candidate_joints_list = [0,1,2,4,5,11,12,13,14,15,16] # =>(27)
    #self.candidate_joints_list = [12,13,14,15,16] # =>(27)
    self.candidate_joints_mum = len(self.candidate_joints_list)
    self.max_mask_joints = 2

    self.upper_body_mask_without_hips =  tf.concat( [tf.zeros(shape=(7,),dtype=tf.float32),tf.ones(shape=(10,),dtype=tf.float32)], axis=0)
    self.upper_body_mask_with_hips = tf.concat( [[1.,1.,0.,0.,1.,0.,0.,], tf.ones(shape=(10,),dtype=tf.float32)], axis=0) 

    'sample_weights'
    self.mask_sample_weight = 2.

  def get_total_chunks(self):
    return self.total_chunks
 
  def get_total_videos(self):
    return self.total_videos

  def get_pairs(self):
    return self.pairs 
  
  def gen(self):

    chunked_poses_2d_list = []
    chunked_poses_3d_list = []
    kps_confidence_list = []
    for (seq_i, start_ind, end_ind, flip) in self.pairs:
      
      '2D poses' 
      seq_2d = self.poses_2d_list[seq_i]
      '3D poses' 
      seq_3d = self.poses_3d_list[seq_i]
      assert seq_2d.shape[0]==seq_3d.shape[0]

      'padding'
      low_ind = tf.maximum(start_ind, 0)
      high_ind = tf.minimum(end_ind, seq_2d.shape[0])
      pad_left = low_ind - start_ind
      pad_right = end_ind - high_ind
      if pad_left != 0 or pad_right != 0:
        chunked_pose_2d = tf.pad( seq_2d[low_ind:high_ind], [[pad_left,pad_right], [0, 0,], [0, 0,]], "CONSTANT")
        chunked_pose_3d = tf.pad( seq_3d[low_ind:high_ind], [[pad_left,pad_right], [0, 0,], [0, 0,]], "CONSTANT")
        #conf  = 
      else:
        chunked_pose_2d = seq_2d[low_ind:high_ind]
        chunked_pose_3d = seq_3d[low_ind:high_ind]

      'give confidence for poseformer'
      cond = tf.not_equal(tf.reduce_sum(chunked_pose_3d, axis=-1),0.) #(27,17,3)=>(27,17)
      kps_confidence = tf.where(cond,1.,0.) #(27,17)

      #tf.reduce_all(tf.equal(chunked_pose_2d,0)

      'Cameras'
      if self.cameras is not None:
        seq_cam = self.cameras[seq_i]
  
      #yield chunked_pose_2d, chunked_pose_3d, seq_cam
      chunked_poses_2d_list.append(chunked_pose_2d)
      chunked_poses_3d_list.append(chunked_pose_3d)
      kps_confidence_list.append(kps_confidence)

    #dataset = tf.data.Dataset.from_tensor_slices((chunked_poses_2d_list, chunked_poses_3d_list, kps_confidence_list))
    return [chunked_poses_2d_list,chunked_poses_3d_list,kps_confidence_list]

  def random_joints_mask(self) :
  
    random_indices = tf.random.uniform(shape=(self.gt,),
                      maxval=self.candidate_joints_mum,
                      dtype=tf.int32)
  
    mask_joints = tf.gather(self.candidate_joints_list,random_indices)
   
    indices = tf.stack([self.gt_ids,mask_joints], axis=-1)
    updates = tf.ones(shape=(self.gt,))
    shape = tf.constant([self.gt,self.kps_num])
    scatter = tf.scatter_nd(indices, updates, shape)
    mask_map = tf.where(tf.cast(scatter,tf.bool),0.,1.)
    return mask_map



  def random_flip(self, chunked_pose_2d, chunked_pose_3d, kps_confidence_GTx17) :
    # Flip 2D keypoints
    index_flip = [0,4,5,6,1,2,3,7,8,9,10,14,15,16,11,12,13]
    flip_chunked_pose_2d = tf.stack([chunked_pose_2d[...,0]*-1., chunked_pose_2d[...,1]], axis=-1)
    flip_chunked_pose_2d = tf.gather(flip_chunked_pose_2d, index_flip, axis=1)

    flip_chunked_pose_3d = tf.concat([chunked_pose_3d[...,0:1]*-1., chunked_pose_3d[...,1:3]], axis=-1)
    flip_chunked_pose_3d = tf.gather(flip_chunked_pose_3d, index_flip, axis=1)

    kps_confidence_GTx17 = tf.gather(kps_confidence_GTx17, index_flip, axis=1)

    return flip_chunked_pose_2d, flip_chunked_pose_3d, kps_confidence_GTx17

  def argumenet_fun(self,
            chunked_pose_2d, 
            chunked_pose_3d, 
            kps_confidence_GTx17):
    """
    chunked_poses_2d : (27,17,2)
    chunked_poses_3d : (27,17,3)
    """
    'flip'
    if tf.random.uniform(())>0.5 :
      chunked_pose_2d, chunked_pose_3d, kps_confidence_GTx17 = self.random_flip(chunked_pose_2d,chunked_pose_3d,kps_confidence_GTx17)

    kps_confidence = kps_confidence_GTx17[self.gt//2,:] # (17,)
    #kps_confidence = kps_confidence_GTx17 #17
    sample_weights = kps_confidence_GTx17


     
    'upper body'
    if self.upper_body_prob > tf.random.uniform(()):

      upper_bbody_mask = tf.where(tf.greater(tf.random.uniform(()),0.5),
                      self.upper_body_mask_without_hips,
                      self.upper_body_mask_with_hips
                      ) #(17,)
                    
      chunked_pose_2d = chunked_pose_2d*upper_bbody_mask[None,:,None] #(27,17,2)
      chunked_pose_3d = chunked_pose_3d*upper_bbody_mask[None,:,None]
      kps_confidence = kps_confidence*upper_bbody_mask #(17,)
      sample_weights *= upper_bbody_mask #(27,17)
    
    balence_weights = tf.constant([1.4, 1.4, 2., 2., 1.4, 2., 2. ], dtype=tf.float32)
    balence_weights =  tf.concat( [balence_weights, tf.ones(shape=(10,),dtype=tf.float32)], axis=0)
    sample_weights *= balence_weights
     
    'joint mask'
    if self.joints_mask_prob> tf.random.uniform(()):
      mask_map = self.random_joints_mask() #(27,17)
      chunked_pose_2d = mask_map[:,:,None]*chunked_pose_2d
      sample_weights *= tf.where(tf.equal(mask_map,0.),self.mask_sample_weight,1.)


    return chunked_pose_2d, chunked_pose_3d, sample_weights

  
  def build(self, batch_Size, test_mode=False):
    'core of training pipeline '
    chunked_poses_2d_list, chunked_poses_3d_list, kps_confidence_GTx17  = self.gen()

    """
    chunked_poses_2d_list : [(27,17,2),....]
    chunked_poses_3d_list : [(27,17,2),....]
    kps_confidence_GTx17 : [(27,17),....]
    """
    autotune = tf.data.AUTOTUNE
    chunked_dataset = tf.data.Dataset.from_tensor_slices((chunked_poses_2d_list, chunked_poses_3d_list, kps_confidence_GTx17))
    if test_mode :
      chunked_dataset = chunked_dataset.cache().shuffle(batch_Size*4).map(self.argumenet_fun, num_parallel_calls=autotune)
    else:
      chunked_dataset = chunked_dataset.cache().shuffle(batch_Size*4).map(self.argumenet_fun, num_parallel_calls=autotune).repeat()
    batch_dataset = chunked_dataset.batch(batch_Size).prefetch(batch_Size*2)
    return batch_dataset

