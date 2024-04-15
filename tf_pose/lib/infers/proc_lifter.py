
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from tensorflow import Tensor
import tensorflow as tf
from tensorflow.keras.layers import  Input
from tensorflow.keras.models import Model
from .base import BaseProc
from lib.Registers import INFER_PROC



# @INFER_PROC.register_module()
# class PreProcess_Lifter(BaseProc):
#     version = '1.0.0'
#     r""" PreProcess_Lifter

#     """
#     def __init__(self, 
#                 num_joints : int=17,
#                 kps2d_input_dims : int = 3,
#                 frames : int = 27,
#                 cam_res_wh : Tuple[int] = (1000, 1000),
#                 coco2h36m_conversion : bool = True,
#                 normalize_kps_2d : bool = True,
#                 batch_size :int = 1,
#                 name=None ):
        
#         self.kps2d_input_dims = kps2d_input_dims
#         self.normalize = normalize_kps_2d

       
#         self.cam_res_wh = tf.constant([[cam_res_wh[0], cam_res_wh[1]]], dtype=tf.float32)
#         if batch_size != 1:  
#            self.cam_res_wh =  tf.tile(self.cam_res_wh,(batch_size,1))

#         self.num_joints = num_joints
#         self.frames = frames
#         self.coco2h36m_conversion = coco2h36m_conversion


#         if 0 :
#             print("\n\n ======= PoseDet Model CFG  -----< PreProcess > =============")
#             print(f"Model input shape : {self.img_shape_yx} \n\n")
#             print(f"use_udp  : {self.use_udp} \n\n")
#             print(f"keep_aspect  : {self.keep_aspect} \n\n")

#         super().__init__(name=name,  
#                         batch_size = batch_size)
        
    
#     def normalize_screen_coordinates(self, X, w, h): 
#         '''
#         x : (b,17,2)
#         w :(b,)
#         h :(b,)
#         '''
#         assert X.shape[-1] == 2
#         #w = w[...,None,None]  #(b,) -> (b,1,1)
#         #h = h[...,None,None]  #(b,) -> (b,1,1)
#         # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
#         return X/w*2 - [1, h/w]
    
#     def tf_coco2h36m_convert_keypoint_definition(self, coco_kps):
#         '''
#         h36m :  1,  2,  3, 4,   5,  6, 9, 11, 12, 13, 14, 15, 16
#         coco : 12, 14, 16, 11, 13, 15, 0,  5,  7,  9,  6,  8, 10
#         '''
#         assert coco_kps.shape[1]==17, \
#         'only support 17kps @tf_coco2h36m_convert_keypoint_definition'
#         coco_kps = coco_kps[...,:2] #(b,17, 2)
#         kp_0 =  (coco_kps[:,11:12,:] + coco_kps[:,12:13,:]) / 2 #(b,1,2)
#         kp_8 =  (coco_kps[:,5:6,:] + coco_kps[:,6:7,:]) / 2
#         kp_7 =  (kp_0 + kp_8) / 2 #(b,1, 2)
#         kp_10 = (coco_kps[:,1:2,:] + coco_kps[:,2:3,:]) / 2

#         kp_1_to_6 = tf.gather(params=coco_kps,indices=[12, 14, 16, 11, 13, 15],axis=1)
#         kp_9 = coco_kps[:,0:1,:]
#         kp_11_to_16 = tf.gather(params=coco_kps,indices=[5,  7,  9,  6,  8, 10],axis=1)

#         h36m_keypoints = tf.concat([kp_0, kp_1_to_6, kp_7, kp_8, kp_9, kp_10, kp_11_to_16], axis=1) #(b,17,2)

#         return h36m_keypoints
    

#     def Set_InputsTensorSpec(self, batch_size): 
#         """Set_InputsTensorSpec.
#         Args:
#             x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
#                 torch.Tensor, containing input data for forward computation.
#         Note:
#             two inputs, one is iamge and other is bbox_yx
#         """
    

#         curr_kps = Input(shape=(self.num_joints, self.kps2d_input_dims), batch_size=batch_size)
#         buffer_kps = Input(shape=(self.frames,self.num_joints, 2), batch_size=batch_size)
#         cam_res_wh = Input(shape=(2,), batch_size=batch_size)

#         InputsTensor = [ curr_kps, buffer_kps, cam_res_wh]

#         return InputsTensor
    
#     def forward(self, 
#                 x : Union[List[Tensor],Tuple[Tensor]]) -> Union[List[Tensor],Tuple[Tensor]]:
        
#         assert isinstance(x,List),"input format must be list | Tuple  @PreProcess_Lifter.forward"
        
  
#         coco_kps_curr, buffer_h36m_norm_kps_prev, cam_res_wh = x

#         """
#         coco_kps_curr : (b, 17,3)
#         buffer_h36m_norm_kps_prev : (b,f,17,2)
#         cam_res_wh : (b,2)
#         """
#         if self.coco2h36m_conversion :
#             h36m_kps_curr = self.tf_coco2h36m_convert_keypoint_definition(coco_kps_curr) #(1,17,3) => (1,17,2)
#         else:
#             h36m_kps_curr = coco_kps_curr[...,:2] #(1,17,3) => (1,17,2)

#         if self.normalize :
#             h36m_norm_kps_curr = self.normalize_screen_coordinates(h36m_kps_curr, cam_res_wh[0,0], cam_res_wh[0,1]) #(1,17,2)
#         else:
#             h36m_norm_kps_curr = h36m_kps_curr
            
#         buffer_h36m_norm_kps_curr = tf.concat( [ buffer_h36m_norm_kps_prev[:,1:,:,:], h36m_norm_kps_curr[:,None,...]],axis=1)

#         """
#         buffer_h36m_norm_kps_curr : (b, f,17,2)
#         """      
#         return  buffer_h36m_norm_kps_curr
    
#     @tf.function(jit_compile=True)
#     def __call__(self, 
#                 coco_kps_curr : Tensor,
#                 buffer_h36m_norm_kps_prev : Tensor ) ->Tuple[Tensor]:

#         assert coco_kps_curr.shape.rank==2 or coco_kps_curr.shape.rank==3,\
#         f"shape.rank of input data must be 2 or 3, but got {coco_kps_curr.shape} @{self.__calss__.__name__}"

#         assert buffer_h36m_norm_kps_prev.shape.rank==3 or buffer_h36m_norm_kps_prev.shape.rank==4,\
#         f"buffer_h36m_norm_kps_prev must be 3 or 4, but got {buffer_h36m_norm_kps_prev.shape} @{self.__calss__.__name__}"

#         if coco_kps_curr.shape.rank==2:
#             coco_kps_curr = tf.expand_dims(coco_kps_curr, axis=0)

#         if buffer_h36m_norm_kps_prev.shape.rank==3:
#             buffer_h36m_norm_kps_prev = tf.expand_dims(buffer_h36m_norm_kps_prev, axis=0)

#         return self.model([coco_kps_curr,buffer_h36m_norm_kps_prev, self.cam_res_wh])   
    


@INFER_PROC.register_module()
class PreProcess_Lifter(BaseProc):
    version = '2.0.0'
    r""" PreProcess_Lifter
    support buffer operation in GPU, 

    """
    def __init__(self, 
                num_joints : int=17,
                kps2d_input_dims : int = 3,
                frames : int = 27,
                cam_res_wh : Tuple[int] = (1000, 1000),
                coco2h36m_conversion : bool = True,
                normalize_kps_2d : bool = True,
                batch_size :int = 1,
                name=None ):
        
        self.kps2d_input_dims = kps2d_input_dims
        self.normalize = normalize_kps_2d

       
        self.cam_res_wh = tf.constant([[cam_res_wh[0], cam_res_wh[1]]], dtype=tf.float32)
        if batch_size != 1:  
           self.cam_res_wh =  tf.tile(self.cam_res_wh,(batch_size,1))

        self.num_joints = num_joints
        self.frames = frames
        self.coco2h36m_conversion = coco2h36m_conversion

        'tf buffer'
        self.Var_Buffer_KPS = tf.Variable(tf.zeros(shape=(batch_size, self.frames, num_joints, 2)), dtype=tf.float32)

        if 0 :
            print("\n\n ======= PoseDet Model CFG  -----< PreProcess > =============")
            print(f"Model input shape : {self.img_shape_yx} \n\n")
            print(f"use_udp  : {self.use_udp} \n\n")
            print(f"keep_aspect  : {self.keep_aspect} \n\n")

        super().__init__(name=name,  
                        batch_size = batch_size)
        
    
    def normalize_screen_coordinates(self, X, w, h): 
        '''
        x : (b,17,2)
        w :(b,)
        h :(b,)
        '''
        assert X.shape[-1] == 2
        #w = w[...,None,None]  #(b,) -> (b,1,1)
        #h = h[...,None,None]  #(b,) -> (b,1,1)
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return X/w*2 - [1, h/w]
    
    def tf_coco2h36m_convert_keypoint_definition(self, coco_kps):
        '''
        h36m :  1,  2,  3, 4,   5,  6, 9, 11, 12, 13, 14, 15, 16
        coco : 12, 14, 16, 11, 13, 15, 0,  5,  7,  9,  6,  8, 10
        '''
        assert coco_kps.shape[1]==17, \
        'only support 17kps @tf_coco2h36m_convert_keypoint_definition'
        coco_kps = coco_kps[...,:2] #(b,17, 2)
        kp_0 =  (coco_kps[:,11:12,:] + coco_kps[:,12:13,:]) / 2 #(b,1,2)
        kp_8 =  (coco_kps[:,5:6,:] + coco_kps[:,6:7,:]) / 2
        kp_7 =  (kp_0 + kp_8) / 2 #(b,1, 2)
        kp_10 = (coco_kps[:,1:2,:] + coco_kps[:,2:3,:]) / 2

        kp_1_to_6 = tf.gather(params=coco_kps,indices=[12, 14, 16, 11, 13, 15],axis=1)
        kp_9 = coco_kps[:,0:1,:]
        kp_11_to_16 = tf.gather(params=coco_kps,indices=[5,  7,  9,  6,  8, 10],axis=1)

        h36m_keypoints = tf.concat([kp_0, kp_1_to_6, kp_7, kp_8, kp_9, kp_10, kp_11_to_16], axis=1) #(b,17,2)

        return h36m_keypoints
    

    def Set_InputsTensorSpec(self, batch_size): 
        """Set_InputsTensorSpec.
        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        Note:
            two inputs, one is iamge and other is bbox_yx
        """
    
        curr_kps = Input(shape=(self.num_joints, self.kps2d_input_dims), batch_size=batch_size)
        cam_res_wh = Input(shape=(2,), batch_size=batch_size)
        InputsTensor = [ curr_kps, cam_res_wh]
        return InputsTensor
    
    def forward(self, 
                x : Union[List[Tensor],Tuple[Tensor]]) -> Union[List[Tensor],Tuple[Tensor]]:
        
        assert isinstance(x,List),"input format must be list | Tuple  @PreProcess_Lifter.forward"
        
  
        coco_kps_curr, cam_res_wh = x

        """
        coco_kps_curr : (b, 17,3)
        buffer_h36m_norm_kps_prev : (b,f,17,2)
        cam_res_wh : (b,2)
        """
        if self.coco2h36m_conversion :
            h36m_kps_curr = self.tf_coco2h36m_convert_keypoint_definition(coco_kps_curr) #(1,17,3) => (1,17,2)
        else:
            h36m_kps_curr = coco_kps_curr[...,:2] #(1,17,3) => (1,17,2)

        if self.normalize :
            h36m_norm_kps_curr = self.normalize_screen_coordinates(h36m_kps_curr, cam_res_wh[0,0], cam_res_wh[0,1]) #(1,17,2)
        else:
            h36m_norm_kps_curr = h36m_kps_curr

        # self.Var_Buffer_KPS[:, :-1,:,:].assign(self.Var_Buffer_KPS[:,1:,:,:])
        # self.Var_Buffer_KPS[:,-1,:,:].assign(h36m_norm_kps_curr)
  
        #buffer_h36m_norm_kps_curr = tf.concat( [ buffer_h36m_norm_kps_prev[:,1:,:,:], h36m_norm_kps_curr[:,None,...]],axis=1)
        # buffer_h36m_norm_kps_curr =  self.Var_Buffer_KPS

        """
        buffer_h36m_norm_kps_curr : (b, f,17,2)
        """      
        return  h36m_norm_kps_curr
    
    @tf.function(jit_compile=True)
    def __call__(self, 
                coco_kps_curr : Tensor) ->Tuple[Tensor]:

        assert coco_kps_curr.shape.rank==2 or coco_kps_curr.shape.rank==3,\
        f"shape.rank of input data must be 2 or 3, but got {coco_kps_curr.shape} @{self.__calss__.__name__}"

        if coco_kps_curr.shape.rank==2:
            coco_kps_curr = tf.expand_dims(coco_kps_curr, axis=0)

        # assert buffer_h36m_norm_kps_prev.shape.rank==3 or buffer_h36m_norm_kps_prev.shape.rank==4,\
        # f"buffer_h36m_norm_kps_prev must be 3 or 4, but got {buffer_h36m_norm_kps_prev.shape} @{self.__calss__.__name__}"
        # if buffer_h36m_norm_kps_prev.shape.rank==3:
        #     buffer_h36m_norm_kps_prev = tf.expand_dims(buffer_h36m_norm_kps_prev, axis=0)


        h36m_norm_kps_curr = self.model([coco_kps_curr, self.cam_res_wh]) 
        self.Var_Buffer_KPS[:, :-1,:,:].assign(self.Var_Buffer_KPS[:,1:,:,:])
        self.Var_Buffer_KPS[:,-1,:,:].assign(h36m_norm_kps_curr)
        buffer_h36m_norm_kps_curr =  self.Var_Buffer_KPS

        return buffer_h36m_norm_kps_curr








@INFER_PROC.register_module()
class PostProcess_Lifter(BaseProc):
    version = '1.0.0'
    r""" PreProcess_Lifter

    """
    def __init__(self, 
                num_joints : int=17,
                frames : int = 27,
                cam_orientation : List[float] = [0., 0., -0.71, 0.71],
                cam_translation : List[float] = [0., 0., 0.],
                Pose3D_trajectory_Enable : bool = True,
                batch_size :int = 1,
                name=None ):
        
        
        self.cam_quat = tf.constant( [cam_orientation], dtype=tf.float32) #(1,4)
        self.cam_tran = tf.constant( [cam_translation], dtype=tf.float32) #(1,3)
        self.num_joints = num_joints
        self.frames = frames
        self.Pose3D_trajectory_Enable = Pose3D_trajectory_Enable

        if not self.Pose3D_trajectory_Enable :
            self.Pose3D_zero_trajectory = tf.zeros(shape=(batch_size, self.frames,3), dtype=tf.float32)

        if 0 :
            print("\n\n ======= PoseDet Model CFG  -----< PreProcess > =============")
            print(f"Model input shape : {self.img_shape_yx} \n\n")
            print(f"use_udp  : {self.use_udp} \n\n")
            print(f"keep_aspect  : {self.keep_aspect} \n\n")

        super().__init__(name=name,  
                        batch_size = batch_size)
        
    def image_coordinates(self, X, w, h):
        assert X.shape[-1] == 2 
        # Reverse camera frame normalization
        return (X + [1, h/w])*w/2    

    def tf_camera_to_world(self, Vec, quat, trans):

        q0 = tf.cast( quat[0], dtype=tf.float32)
        q1 = tf.cast( quat[1], dtype=tf.float32)
        q2 = tf.cast( quat[2], dtype=tf.float32)
        q3 = tf.cast( quat[3], dtype=tf.float32)

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

        Vec_Out = tf.linalg.matmul(a=Rot[None,None,None, :,:],b=Vec[...,None])*2.0
        Vec_Out = tf.squeeze(Vec_Out, axis=-1)
        Vec_Out = tf.cast( Vec_Out + trans, dtype=tf.float32)
        return Vec_Out
    

    def Set_InputsTensorSpec(self, batch_size): 
        """Set_InputsTensorSpec.
        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        Note:
            two inputs, one is iamge and other is bbox_yx
        """
        Pose3D_pred_cam  = Input(shape=(self.frames , self.num_joints, 3), batch_size=batch_size)
        trajectory_cam = Input(shape=(self.frames , 3), batch_size=batch_size) 
        quat = Input(shape=(4,), batch_size=batch_size) # Quaternion (qw,qx,qy,qz)
        trans = Input(shape=(3,), batch_size=batch_size) # transition (x,y,z)


        InputsTensor = [ Pose3D_pred_cam, trajectory_cam, quat, trans]

        return InputsTensor
    
    def forward(self, 
                x : Union[List[Tensor],Tuple[Tensor]]) -> Union[List[Tensor],Tuple[Tensor]]:
        
      #===================================================================================
        # Pose3D_pred : (1,f,17,3)  
        # Pose3D_trajectory_cam:(1,f,3)  3=(x,y,z)
        # cam_res_wh : (1, 2)
        # cam_orientation : (1, 4)
        # cam_translation : (1, 3)
        #===================================================================================

        Pose3D_pred_cam, Pose3D_trajectory_cam, cam_quat , cam_tran = x

        cond = tf.greater(tf.math.count_nonzero(Pose3D_trajectory_cam),0) #(1,f,3) 
        Pose3D_pred_cam = Pose3D_pred_cam  #(1,f,17,3)
        #Pose3D_pred_cam = tf.squeeze(Pose3D_pred_cam,axis=1) #(1,f, 17,3)

 
        Pose3D_pred_cam += Pose3D_trajectory_cam[:,:,None,:]   #(1,f, 17,3) + (1,f,3) = (1,f,17,3)
        
        Pose3D_pred = self.tf_camera_to_world(tf.cast(Pose3D_pred_cam, dtype=tf.float32), 
                                                quat  = cam_quat[0], 
                                                trans = cam_tran[0])   #(1,f,17,3)
        
        rebase_height =tf.where(cond, Pose3D_pred[...,2:3], Pose3D_pred[...,2:3] -tf.reduce_min(Pose3D_pred[..., 2]) ) #(1,f, 17,1)
        Pose3D_pred = tf.concat([Pose3D_pred[...,:2], rebase_height], axis=-1) #(1,17,3)

        """
   
        """      
        return  Pose3D_pred
    
    @tf.function(jit_compile=True)
    def __call__(self, 
                Pose3D_pred_cam : Tensor,
                Pose3D_trajectory_cam : Optional[Tensor]=None ) ->Tensor:
        '''
        Pose3D_pred : (1,f,17,3)  
        Pose3D_trajectory_cam:(1,f,3)  3=(x,y,z)
        '''
        '''
        assert feats_list[0].shape.rank==5 and len(feats_list)==3, \
        "shape.rank of feats must be 5 @PostProcess_YOLO_Det.__call__"

        assert meta.shape.rank==2, \
        "shape.rank of meta must be 2 @PostProcess_YOLO_Det.__call__"
        '''
        #
        if self.Pose3D_trajectory_Enable == False or Pose3D_trajectory_cam is None:
           Pose3D_trajectory_cam = self.Pose3D_zero_trajectory 
        else:
            assert Pose3D_trajectory_cam.shape.rank==2 or Pose3D_trajectory_cam.shape.rank==3, \
            f"Pose3D_trajectory_cam must be 2 or 3, but got {Pose3D_trajectory_cam.shape} @{self.__calss__.__name__}"
            if Pose3D_trajectory_cam.shape.rank==2:
                Pose3D_trajectory_cam = tf.exapnd_dims(Pose3D_trajectory_cam, axis=0)

        
        return self.model([Pose3D_pred_cam, Pose3D_trajectory_cam, self.cam_quat, self.cam_tran])   