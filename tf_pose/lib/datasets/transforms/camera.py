from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import tensorflow as tf

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
def normalize_screen_coordinates(X, w, h): 
  assert X.shape[-1] == 2
  # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
  return X/w*2 - [1, h/w]

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
def image_coordinates(X, w, h):
  assert X.shape[-1] == 2 
  # Reverse camera frame normalization
  return (X + [1, h/w])*w/2

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
def tf_world_to_camera(Vec, quat, trans):

    #assert quaternion.shape[-1] == 4
    #assert Vec.shape[-1] == 3
    Vec = tf.cast(Vec, dtype=tf.float64)
    trans = tf.cast(trans, dtype=tf.float64)

    q0 = tf.cast( quat[0], dtype=tf.float64)
    q1 = tf.cast( quat[1], dtype=tf.float64)
    q2 = tf.cast( quat[2], dtype=tf.float64)
    q3 = tf.cast( quat[3], dtype=tf.float64)

    R11 = (0.5-q2*q2-q3*q3)
    R12 = (q0*q3 + q1*q2)
    R13 = (q1*q3 - q0*q2) 

    R21 = (q1*q2 - q0*q3)
    R22 = (0.5-q1*q1-q3*q3)
    R23 = (q0*q1 + q2*q3)

    R31 = (q0*q2 + q1*q3)
    R32 = (q2*q3 - q0*q1)
    R33 = (0.5-q1*q1-q2*q2)

    Rot = tf.stack([R11,R12,R13,R21,R22,R23,R31,R32,R33])
    Rot = tf.reshape(Rot,(3,3))

    Vec_Out = Vec - trans
    Vec_Out = tf.linalg.matmul(a=Rot[None,None,:,:],b=Vec_Out[...,None])*2.0
    Vec_Out = tf.cast(tf.squeeze(Vec_Out),dtype=tf.float32)
    return Vec_Out.numpy()

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
def tf_camera_to_world(Vec, quat, trans):

    q0 = tf.cast( quat[0], dtype=tf.float64)
    q1 = tf.cast( quat[1], dtype=tf.float64)
    q2 = tf.cast( quat[2], dtype=tf.float64)
    q3 = tf.cast( quat[3], dtype=tf.float64)

    R11 = (0.5-q2*q2-q3*q3)
    R12 = (-q0*q3 + q1*q2)
    R13 = (q1*q3 + q0*q2) 

    R21 = (q1*q2 + q0*q3)
    R22 = (0.5-q1*q1-q3*q3)
    R23 = (-q0*q1 + q2*q3)

    R31 = (q0*q2 + q1*q3) 
    R32 = (q2*q3 + q0*q1)
    R33 = (0.5-q1*q1-q2*q2)
    
    Rot = tf.stack([R11,R12,R13,R21,R22,R23,R31,R32,R33])
    Rot = tf.reshape(Rot,(3,3))

    Vec_Out = tf.linalg.matmul(a=Rot[None,None,:,:],b=Vec[...,None])*2.0
    Vec_Out = tf.squeeze(Vec_Out)
    Vec_Out = tf.cast( Vec_Out + trans, dtype=tf.float32)
    return Vec_Out.numpy()
