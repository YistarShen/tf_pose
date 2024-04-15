from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import sys, os, cv2
import numpy as np

from tensorflow import Tensor
import tensorflow as tf
from matplotlib import rc, rcParams
rc('animation', html='jshtml')
import matplotlib.pyplot as plt
import ipywidgets, imageio
from ipywidgets import IntProgress
from IPython.display import display
from lib.utils.common import is_path_available, convert_images_into_np_array

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
class TopDownPoseDraw_cocoStyle():
    VERSION = '1.0.0'
    r"""CocoStyle_TopDownPoseDraw
    
    """
    def __init__(self, 
            num_joints = 17,
            show_joints = True,
            hm_thr = tf.ones(shape=(17,), dtype=tf.float32)*0.5,   
            img_type = "all_parts",
            ):
        
        if  num_joints != 17:
            raise NotImplementedError("only support num_joints=17"
                                      f"but got {num_joints}"                         
            )
        
        self.num_joints = num_joints
        self.plot_joints = show_joints

        self.skeletons = [  [5,6],[6,12],[5,11],[11,12],   # face
                            [8,6],[10,8],[5,7],[7,9],      # body
                            [12,14],[14,16],[11,13],[13,15],  #hand and leg
                            [15,17],[15,18],[17,18], [16,19],[16,20],[19,20],  # foot 
                            [9,21],[9,22],[9,23], [22,23] ,    # 9 - 21, 22, 23, left-plam
                            [10,24],[10,25],[10,26], [25,26]     # 10 - 24, 25, 26 right-plam
                        ]
        
        self.right_parts = [2,4,6,8,10,12,14,16,19,20,24,25,26]  
        self.skeletons = self.skeletons[:12]if num_joints == 17 else self.skeletons
        self.right_parts = self.right_parts[:8]if num_joints == 17 else self.right_parts  

        self.img_type = img_type
        if((img_type!="all_parts") and (img_type!="only_right_parts") and img_type!="only_left_parts"):
            raise RuntimeError('unknown img_type : type must be "only_right_parts" or "ceonly_left_partsnters" of "all_parts"') 
    
        'keypoint hr'
        if  isinstance(hm_thr, List):
            self.hm_thr = tf.constant(hm_thr, dtype=tf.float32)
        elif isinstance(hm_thr, tf.Tensor):
            self.hm_thr = hm_thr
        else:
            self.hm_thr = tf.ones(shape=(self.num_joints,), dtype=tf.float32)*0.5
            
        assert self.hm_thr.shape[0]==self.num_joints, \
        f"hm_thr shape : {self.hm_thr.shape} didn't match num_joints : {self.num_joints}"

    def __call__(self,
                kps, 
                bbox_yx,                                    
                img_out_np,
                circle_radius = 20,
                line_width = 15) ->np.ndarray:

        if img_out_np.ndim!=3:
            raise ValueError("img_out_np.ndim must be 3"
                            f" but got {img_out_np.ndim} @{self.__class__.__name__}") 

        if kps is not None and (kps.ndim!=2 or kps.shape!=(self.num_joints,3)):
            raise ValueError("kps.shape must be (17,3)"
                            f" but got {kps.shape} @{self.__class__.__name__}")  

        'show img with predicted bbox'
        top, left, bottom, right=  bbox_yx.numpy()
        cv2.rectangle(img_out_np, 
                    (int(left), int(top)), (int(right), int(bottom)), 
                    color = (255,0,0), 
                    thickness = line_width
        )
        
        if self.plot_joints != True:
            return img_out_np 
                         
        'plot key point on predicted image frame'
        kps_out = []
        for i in range(self.num_joints):
            if(kps[i,2]>self.hm_thr[i]):
                kps_pred_x = ( kps[i,0] )
                kps_pred_y = ( kps[i,1] )
                kps_out.append(tf.stack([kps_pred_x, kps_pred_y, kps[i,2] ], axis=-1))
                'plot key point on input image frame'
                cv2.circle(img_out_np,(int(kps_pred_x), int(kps_pred_y)), circle_radius, (255, 255, 255), -1)
                if i in self.right_parts :
                    cv2.circle(img_out_np,(int(kps_pred_x), int(kps_pred_y)), (circle_radius//3)*2, (0, 255, 255), -1)
                else:
                    cv2.circle(img_out_np,(int(kps_pred_x), int(kps_pred_y)), (circle_radius//3)*2, (255, 140, 0), -1)  
            else:
                kps_out.append([0., 0., kps[i,2] ])  
    
        kps_out = tf.stack(kps_out, axis=0) 
        'plot skeletons (option)'
        for skeleton in self.skeletons :           
            idx_1 = skeleton[0]
            idx_2 = skeleton[1]
            kp_1 = kps_out[idx_1, : ]  
            kp_2 = kps_out[idx_2, : ] 
                
            cond = (self.img_type=="all_parts")|(self.img_type=="only_right_parts" and  (idx_1 in self.right_parts)  and (idx_2 in self.right_parts))
            cond = cond | (self.img_type=="only_left_parts" and  (idx_1 not in self.right_parts)  and (idx_2 not in self.right_parts))
            if(kp_1[2]>self.hm_thr[idx_1] and kp_2[2]>self.hm_thr[idx_2] and cond):
                pt_1 = tf.cast(kps_out[idx_1, : 2], dtype=tf.int32).numpy() 
                pt_2 = tf.cast(kps_out[idx_2, : 2], dtype=tf.int32).numpy()
                cv2.line(img_out_np, pt_1, pt_2, (255, 255, 255), line_width//2) 
        return img_out_np

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
def TopDownPoseImagesPlot_cocoStyle(                
                upload : List[str],
                InferDet :  Optional[Callable] = None,
                InferPose :  Optional[Callable] = None,
                InferTopdown : Optional[Callable] = None,
                num_joints : int = 17,
                hm_thr = tf.ones(shape=(17,), dtype=tf.float32)*0.5,  
                show_joints = True,
                img_type = "all_parts"):
    
    if InferTopdown is not None :
        assert isinstance(InferTopdown,Callable), \
        "InferTopdown is not Callable @TopDownPoseImagesPlot_cocoStyle"
    else:
        if not isinstance(InferDet,Callable) and  not isinstance(InferPose,Callable):
            raise TypeError("InferDet and InferPose must both be Callable type if InferTopdown=None"
                            f"but got InferDet : {type(InferDet)} and InferPose : {type(InferPose)}"
            )


    # if InferDet is not None  and InferPose is not None :
    #     assert isinstance(InferDet,Callable) and isinstance(InferPose,Callable), \
    #     "InferDet or InferPose is not Callable @topdown_PoseDraw_coco_style"
        
    'convert image to np array for image plotting by cv2 lib'
    image_np_list = convert_images_into_np_array(upload)

    'pose draw'
    PoseDraw = TopDownPoseDraw_cocoStyle(num_joints = num_joints, 
                                         show_joints = show_joints, 
                                         hm_thr = hm_thr, 
                                         img_type = img_type)

    kps_preds = []
    bbox_preds = []
    for src_image_np in image_np_list :
        src_image = tf.constant(src_image_np, dtype=tf.uint8)
        src_image =  tf.expand_dims(src_image, axis=0)
        'Inference of Single Person detection --------------------------------START'
        if InferTopdown is None :
            det_dict = InferDet(src_image)
            bbox_yx_SrcImg = det_dict['bbox_yx']
            pose_dict = InferPose(src_image, bbox_yx_SrcImg)
            kps_pred_SrcImg = pose_dict['kps']
        else:
            topdon_dict = InferTopdown(src_image)
            bbox_yx_SrcImg = topdon_dict['det']['bbox_yx']
            kps_pred_SrcImg = topdon_dict['pose']['kps']

        'Inference of  TopDown Pose Detection-------------------------------- END'
        bbox_preds.append(bbox_yx_SrcImg)
        kps_preds.append(kps_pred_SrcImg)

        'set circle_radius / line_width'
        circle_radius = np.max((src_image_np.shape[0],src_image_np.shape[1]))
        circle_radius = np.max( (circle_radius//100, 10))
        line_width =  (circle_radius//2)
        'show img with predicted bbox'
            
        _ = PoseDraw(kps_pred_SrcImg[0,:,:], 
                    bbox_yx_SrcImg[0,:],
                    src_image_np,
                    circle_radius = circle_radius,
                    line_width = line_width
        ) 
                                    
        plt.figure(figsize=(10,10))
        plt.title(f'kps with bbox @src_img frame')
        plt.imshow(src_image_np)
        plt.show()
        #im = Image.fromarray(img_out)
        #im.save("filename.jpeg")
        del src_image_np 

    bbox_preds =  tf.concat(bbox_preds, axis=0)
    kps_preds =  tf.concat(kps_preds, axis=0)
    return {"kps" : kps_preds, "bbox_yx" : bbox_preds}


# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
def TopDownPoseRenderAnimation_cocoStyle(
                            video_path,
                            cv2_rot_k : int = 0,
                            max_frames = None,
                            processing_group_frames = 1,
                            fps_out = 30,
                            extra_save_name='',
                            save_dir = './',  
                            save_video = False,
                            InferDet :  Optional[Callable] = None,
                            InferPose :  Optional[Callable] = None,
                            InferTopdown : Optional[Callable] = None,
                            num_joints : int = 17,
                            hm_thr = tf.ones(shape=(17,), dtype=tf.float32)*0.5,  
                            show_joints = True,
                            img_type = "all_parts"):
    
    is_path_available(video_path)

    'load video'
    video_name = os.path.basename(video_path)[:-4]
    cap = cv2.VideoCapture(video_path)

    'video info'
    cv2_rotateCode = [None,
                      cv2.ROTATE_90_CLOCKWISE, 
                      cv2.ROTATE_180, 
                      cv2.ROTATE_90_COUNTERCLOCKWISE]
    cv2_rot_k = cv2_rotateCode[cv2_rot_k]

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    print("==================Video's INFO=================")
    print("Video name :".format(video_name))
    print("Video's Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    print(f"Video's Frame-HxW  : {length}-{height}x{width}-")

    'init for loadde video'
    if max_frames is not None :
        total_procesiing_counts = int(max_frames/processing_group_frames)
    else:
        max_frames = length 
        total_procesiing_counts = int(length/processing_group_frames)

    print(f"total_procesiing_counts : {total_procesiing_counts}")
    '---------------init ptogress bar-------------------------------------'
    bar = IntProgress(layout = ipywidgets.Layout(width='auto', height='40px') )
    bar.max = total_procesiing_counts
    bar.description = '(Init)'
    display(bar)

    '--------------------setup video writer--------------------------------- '
    if save_video :
        'video name & video saved path'
        video_save_name= f'{video_name}_{fps_out}_{extra_save_name}.mp4'

        #save_dir = os.path.abspath(os.path.join(test_videos_dir, os.path.pardir))

        video_save_path = os.path.join(save_dir, video_save_name)
        'setup imageio writer'
        writer = imageio.get_writer(video_save_path, 
                                    fps=fps_out
        )
        'pose draw'
        PoseDraw = TopDownPoseDraw_cocoStyle(num_joints,
                                            show_joints, 
                                            hm_thr, 
                                            img_type
        )

     
    poses_preds_list  : List[np.ndarray] = []
    bboxes_preds_list : List[np.ndarray] = []

    frame_counts = 0
    while(cap.isOpened()):

        frame_counts  = frame_counts  + 1
        '==========bar update========================================='
        bar.value = frame_counts
        bar.description = f'frame:{frame_counts}'
       
        was_read, src_frame_np = cap.read()
        if not was_read or frame_counts==max_frames:
            break

        'skip frame by processing_group_frames'
        if(frame_counts%processing_group_frames!=1 and processing_group_frames!=1):
            continue   

        'videos property setting'
        src_frame_np = cv2.cvtColor(src_frame_np, cv2.COLOR_BGR2RGB)
        if cv2_rot_k!= None :
            src_frame_np = cv2.rotate(src_frame_np, cv2_rot_k)

   
        '=========== TopDown PoseDet Inference===========================START'
        src_image = tf.constant(src_frame_np, dtype=tf.uint8)
        src_image =  tf.expand_dims(src_image, axis=0)


        if InferTopdown is None :
            det_dict = InferDet(src_image)
            pose_dict = InferPose(src_image, det_dict['bbox_yx'])
        else:
            topdown_dict = InferTopdown(src_image)
            det_dict = topdown_dict['det']
            pose_dict = topdown_dict['pose']

        bbox_pred_SrcImg = det_dict['bbox_yx']
        if pose_dict.get('smooth_kps',None) is not None :
            kps_pred_SrcImg = pose_dict['smooth_kps']
        else:
            kps_pred_SrcImg = pose_dict['kps']
       
        'update list'
        poses_preds_list.append(kps_pred_SrcImg[0,...].numpy())
        bboxes_preds_list.append(bbox_pred_SrcImg[0,...].numpy())

        if save_video :
            _ = PoseDraw(kps_pred_SrcImg[0,:,:], 
                        bbox_pred_SrcImg[0,:],
                        src_frame_np,                                     
                        circle_radius = 8,
                        line_width = 5)  
            
            'write img to file'
            writer.append_data(src_frame_np)

        del src_frame_np
        if  frame_counts == max_frames:
            break

    '----------------------bar final update----------------'
    bar.description = f'Done ({frame_counts})'   
    '--------------------close writer---------------------------------------------------------'
    if save_video : writer.close() ; print(f"video_save_path : \n<{video_save_path}>\n") 

    poses_preds = np.stack(poses_preds_list,axis=0)
    bboxes_preds = np.stack(bboxes_preds_list,axis=0)
        
    results = {"kps_xy" : poses_preds, 
               "bboxes_yx" : bboxes_preds}
      
    '------------------save pred results as .npy--------------------------------------------------------'
    # npy_save_name =  f'{video_name}_{extra_save_name}'
    # pose_2d_pred_save_path = os.path.join(save_dir, npy_save_name)
    # np.save(pose_2d_pred_save_path, results)
    # print(f"pose_2d_pred shape : {poses_2d_pred.shape} @{type(poses_2d_pred)}")
    # print(f"video processing finish !!!!!!!! <>") 
    return  results



            
# ############################################################################
# #
# # 
# ############################################################################
# class topdown_PoseDraw_coco_style():
#     VERSION = '1.0.0'
#     r"""topdown_PoseDraw_coco_style

#     """
#     def __init__(self, 
#             show_joints = True,
#             num_joints = 17,
#             hm_thr = tf.ones(shape=(17,), dtype=tf.float32)*0.5,  
#             img_type = "only_left_parts",
#             ):

#         #assert (num_joints==27 or num_joints==17),"num_joints must be 17 or 27 @TopDown_Pose_Conversion" 
#         self.plot_joints = show_joints
#         self.num_joints = num_joints  
#         ''''
#         self.skeletons = [[5,6],[6,12],[5,11],[11,12],   # face
#                     [8,6],[10,8],[5,7],[7,9],      # body
#                     [12,14],[14,16],[11,13],[13,15],  #hand and leg
#                     [15,17],[15,18],[17,18], [16,19],[16,20],[19,20],  # foot 
#                     [9,21],[9,22],[9,23], [22,23] ,    # 9 - 21, 22, 23, left-plam
#                     [10,24],[10,25],[10,26], [25,26]     # 10 - 24, 25, 26 right-plam
#                     ]
#         self.right_parts = [2,4,6,8,10,12,14,16,19,20,24,25,26]  
#         self.skeletons = self.skeletons[:12]if num_joints == 17 else self.skeletons
#         self.right_parts = self.right_parts[:8]if num_joints == 17 else self.right_parts
#         '''
        
#         self.skeletons = [[5,6],[6,12],[5,11],[11,12],   # face
#                         [8,6],[10,8],[5,7],[7,9],      # body
#                         [12,14],[14,16],[11,13],[13,15],  #hand and leg
#                     ]
#         self.right_parts = [2,4,6,8,10,12,14,16]  
#         #self.skeletons = self.skeletons[:12]if num_joints == 17 else self.skeletons
#         #self.right_parts = self.right_parts[:8]if num_joints == 17 else self.right_parts


#         self.img_type = img_type
#         if((img_type!="all_parts") and (img_type!="only_right_parts") and img_type!="only_left_parts"):
#             raise RuntimeError('unknown img_type : type must be "only_right_parts" or "ceonly_left_partsnters" of "all_parts"') 
    
#         'keypoint hr'
#         if  isinstance(hm_thr, List):
#             self.hm_thr = tf.constant(hm_thr, dtype=tf.float32)
#         elif isinstance(hm_thr, tf.Tensor):
#             self.hm_thr = hm_thr
#         else:
#             self.hm_thr = tf.ones(shape=(self.num_joints,), dtype=tf.float32)*0.5

#         assert self.hm_thr.shape[0]==self.num_joints, \
#         f"hm_thr shape : {self.hm_thr.shape} didn't match num_joints : {self.num_joints}"

#     def drawing_with_kps_skeletons(self, 
#                                     kps, 
#                                     bbox_yx, 
#                                     img_out_np,
#                                     circle_radius = 20,
#                                     line_width = 15):
#         """
#         kps_27x3 : (27,3)/ (17,3)/
#         bbox_yx : (4,)
#         img_out_np : (h,w,3),  dtype : np array
#         """
#         assert (kps.ndim==2 and img_out_np.ndim==3),"please check dims of input kps or img_out_np !!!!" 

#         'show img with predicted bbox'
#         top, left, bottom, right=  bbox_yx.numpy()
#         cv2.rectangle(img_out_np, 
#                     (int(left), int(top)), (int(right), int(bottom)), 
#                     color=(255,0,0), thickness=line_width
#                     )
#         if self.plot_joints != True:
#             return img_out_np
     
#         'plot key point on predicted image frame'
#         kps_out = []
#         for i in range(self.num_joints):
#             if(kps[i,2]>self.hm_thr[i]):
#                 kps_pred_x = ( kps[i,0] )
#                 kps_pred_y = ( kps[i,1] )
#                 kps_out.append(tf.stack([kps_pred_x, kps_pred_y, kps[i,2] ], axis=-1))
#                 'plot key point on input image frame'
#                 cv2.circle(img_out_np,(int(kps_pred_x), int(kps_pred_y)), circle_radius, (255, 255, 255), -1)
#                 if i in self.right_parts :
#                     cv2.circle(img_out_np,(int(kps_pred_x), int(kps_pred_y)), (circle_radius//3)*2, (0, 255, 255), -1)
#                 else:
#                     cv2.circle(img_out_np,(int(kps_pred_x), int(kps_pred_y)), (circle_radius//3)*2, (255, 140, 0), -1)  
#             else:
#                 kps_out.append([0., 0., kps[i,2] ])  
    
#         kps_out = tf.stack(kps_out, axis=0) 
#         'plot skeletons (option)'
#         for skeleton in self.skeletons :           
#             idx_1 = skeleton[0]
#             idx_2 = skeleton[1]
#             kp_1 = kps_out[idx_1, : ]  
#             kp_2 = kps_out[idx_2, : ] 
                
#             cond = (self.img_type=="all_parts")|(self.img_type=="only_right_parts" and  (idx_1 in self.right_parts)  and (idx_2 in self.right_parts))
#             cond = cond | (self.img_type=="only_left_parts" and  (idx_1 not in self.right_parts)  and (idx_2 not in self.right_parts))
#             if(kp_1[2]>self.hm_thr[idx_1] and kp_2[2]>self.hm_thr[idx_2] and cond):
#                 pt_1 = tf.cast(kps_out[idx_1, : 2], dtype=tf.int32).numpy() 
#                 pt_2 = tf.cast(kps_out[idx_2, : 2], dtype=tf.int32).numpy()
#                 cv2.line(img_out_np, pt_1, pt_2, (255, 255, 255), line_width//2) 
#         return img_out_np
    
#     def convert_images_into_np_array(self, img_jpg):
#         images_list = []
#         for img_name in img_jpg :
#             image = cv2.imread(img_name)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             image = np.array(image).reshape( ( image.shape[0], image.shape[1], 3)).astype(np.uint8)
#             images_list.append(image.copy())   
#             print(image.shape)
#             del image
#         return images_list
    
    
#     def __call__(self,
#                 upload : List[str],
#                 InferDet :  Optional[Callable] = None,
#                 InferPose :  Optional[Callable] = None,
#                 InferTopdown : Optional[Callable] = None):
        
#         assert ((InferDet is not None and InferPose is not  None) or InferTopdown is not None), \
#         "@topdown_PoseDraw_coco_style.__call__"

#         if InferTopdown is not None :
#             assert isinstance(InferTopdown,Callable), "InferTopdown is not Callable @topdown_PoseDraw_coco_style"

#         if InferDet is not None  and InferPose is not None :
#             assert isinstance(InferDet,Callable) and isinstance(InferPose,Callable), \
#             "InferDet or InferPose is not Callable @topdown_PoseDraw_coco_style"

#         def convert_images_into_np_array(img_jpg):
#             images_list = []
#             for img_name in img_jpg :
#                 image = cv2.imread(img_name)
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 image = np.array(image).reshape( ( image.shape[0], image.shape[1], 3)).astype(np.uint8)
#                 images_list.append(image.copy())   
#                 print(image.shape)
#                 del image
#             return images_list
        
#         'convert image to np array for image plotting by cv2 lib'
#         image_np_list = convert_images_into_np_array(upload)


#         kps_preds = []
#         bbox_preds = []
#         for src_image_np in image_np_list :

#             src_image = tf.constant(src_image_np, dtype=tf.uint8)
#             src_image =  tf.expand_dims(src_image, axis=0)

#             'Inference of Single Person detection --------------------------------START'
#             # if InferTopdown is None :
#             #     bbox_yx_SrcImg = InferDet(src_image)
#             #     kps_pred_SrcImg = InferPose(src_image, bbox_yx_SrcImg)
#             # else:
#             #     kps_pred_SrcImg, bbox_yx_SrcImg = InferTopdown(src_image)

#             if InferTopdown is None :
#                 det_dict = InferDet(src_image)
#                 bbox_yx_SrcImg = det_dict['bbox_yx']
#                 pose_dict = InferPose(src_image, bbox_yx_SrcImg)
#                 kps_pred_SrcImg = pose_dict['kps']
#             else:
#                 topdon_dict = InferTopdown(src_image)
#                 bbox_yx_SrcImg = topdon_dict['det']['bbox_yx']
#                 kps_pred_SrcImg = topdon_dict['pose']['kps']

#             'Inference of  TopDown Pose Detection-------------------------------- END'
#             bbox_preds.append(bbox_yx_SrcImg)
#             kps_preds.append(kps_pred_SrcImg)

#             'set circle_radius / line_width'
#             circle_radius = np.max((src_image_np.shape[0],src_image_np.shape[1]))
#             circle_radius = np.max( (circle_radius//100, 10))
#             line_width =  (circle_radius//2)
#             'show img with predicted bbox'
            
#             self.drawing_with_kps_skeletons(kps_pred_SrcImg[0,:,:], 
#                                             bbox_yx_SrcImg[0,:],
#                                             src_image_np,
#                                             circle_radius = circle_radius,
#                                             line_width = line_width
#             )                             
#             plt.figure(figsize=(10,10))
#             plt.title(f'image with personal bbox @ src_img frame')
#             plt.imshow(src_image_np)
#             plt.show()

#             #im = Image.fromarray(img_out)
#             #im.save("filename.jpeg")
#             del src_image_np 

#         bbox_preds =  tf.concat(bbox_preds, axis=0)
#         kps_preds =  tf.concat(kps_preds, axis=0)
#         return {"kps" : kps_preds, "bbox_yx" : bbox_preds}
    


# ############################################################################
# #
# # 
# ############################################################################
# class coco_17kps_PoseDraw():
#     def __init__(self, 
#             show_joints = True,
#             num_joints =17,
#             hm_thr = None,  
#             img_type = "all_parts",
#             ):

#         #assert (num_joints==27 or num_joints==17),"num_joints must be 17 or 27 @TopDown_Pose_Conversion" 
#         self.plot_joints = show_joints

#         self.num_joints = num_joints  
#         ''''
#         self.skeletons = [[5,6],[6,12],[5,11],[11,12],   # face
#                     [8,6],[10,8],[5,7],[7,9],      # body
#                     [12,14],[14,16],[11,13],[13,15],  #hand and leg
#                     [15,17],[15,18],[17,18], [16,19],[16,20],[19,20],  # foot 
#                     [9,21],[9,22],[9,23], [22,23] ,    # 9 - 21, 22, 23, left-plam
#                     [10,24],[10,25],[10,26], [25,26]     # 10 - 24, 25, 26 right-plam
#                     ]
#         self.right_parts = [2,4,6,8,10,12,14,16,19,20,24,25,26]  
#         self.skeletons = self.skeletons[:12]if num_joints == 17 else self.skeletons
#         self.right_parts = self.right_parts[:8]if num_joints == 17 else self.right_parts
#         '''
        
#         self.skeletons = [[5,6],[6,12],[5,11],[11,12],   # face
#                     [8,6],[10,8],[5,7],[7,9],      # body
#                     [12,14],[14,16],[11,13],[13,15],  #hand and leg
#                     ]
#         self.right_parts = [2,4,6,8,10,12,14,16]  
#         #self.skeletons = self.skeletons[:12]if num_joints == 17 else self.skeletons
#         #self.right_parts = self.right_parts[:8]if num_joints == 17 else self.right_parts


#         self.img_type = img_type
#         if((img_type!="all_parts") and (img_type!="only_right_parts") and img_type!="only_left_parts"):
#             raise RuntimeError('unknown img_type : type must be "only_right_parts" or "ceonly_left_partsnters" of "all_parts"') 
    
#         'keypoint hr'
#         if  isinstance(hm_thr, List):
#             self.hm_thr = tf.constant(hm_thr, dtype=tf.float32)
#         elif isinstance(hm_thr, tf.Tensor):
#             self.hm_thr = hm_thr
#         else:
#             self.hm_thr = tf.ones(shape=(self.num_joints,), dtype=tf.float32)*0.5

#         assert self.hm_thr.shape[0]==self.num_joints, \
#         f"hm_thr shape : {self.hm_thr.shape} didn't match num_joints : {self.num_joints}"

#     def drawing_with_kps_skeletons(self, 
#                                     kps, 
#                                     bbox_yx, 
#                                     img_out_np,
#                                     circle_radius = 20,
#                                     line_width = 15):
#         """
#         kps_27x3 : (27,3)/ (17,3)/
#         bbox_yx : (4,)
#         img_out_np : (h,w,3),  dtype : np array
#         """
#         assert (kps.ndim==2 and img_out_np.ndim==3),"please check dims of input kps or img_out_np !!!!" 

#         'show img with predicted bbox'
#         top, left, bottom, right=  bbox_yx.numpy()
#         cv2.rectangle(img_out_np, 
#                     (int(left), int(top)), (int(right), int(bottom)), 
#                     color=(255,0,0), thickness=line_width
#                     )
#         if self.plot_joints != True:
#             return img_out_np
     
#         'plot key point on predicted image frame'
#         kps_out = []
#         for i in range(self.num_joints):
#             if(kps[i,2]>self.hm_thr[i]):
#                 kps_pred_x = ( kps[i,0] )
#                 kps_pred_y = ( kps[i,1] )
#                 kps_out.append(tf.stack([kps_pred_x, kps_pred_y, kps[i,2] ], axis=-1))
#                 'plot key point on input image frame'
#                 cv2.circle(img_out_np,(int(kps_pred_x), int(kps_pred_y)), circle_radius, (255, 255, 255), -1)
#                 if i in self.right_parts :
#                     cv2.circle(img_out_np,(int(kps_pred_x), int(kps_pred_y)), (circle_radius//3)*2, (0, 255, 255), -1)
#                 else:
#                     cv2.circle(img_out_np,(int(kps_pred_x), int(kps_pred_y)), (circle_radius//3)*2, (255, 140, 0), -1)  
#             else:
#                 kps_out.append([0., 0., kps[i,2] ])  
    
#         kps_out = tf.stack(kps_out, axis=0) 
#         'plot skeletons (option)'
#         for skeleton in self.skeletons :           
#             idx_1 = skeleton[0]
#             idx_2 = skeleton[1]
#             kp_1 = kps_out[idx_1, : ]  
#             kp_2 = kps_out[idx_2, : ] 
                
#             cond = (self.img_type=="all_parts")|(self.img_type=="only_right_parts" and  (idx_1 in self.right_parts)  and (idx_2 in self.right_parts))
#             cond = cond | (self.img_type=="only_left_parts" and  (idx_1 not in self.right_parts)  and (idx_2 not in self.right_parts))
#             if(kp_1[2]>self.hm_thr[idx_1] and kp_2[2]>self.hm_thr[idx_2] and cond):
#                 pt_1 = tf.cast(kps_out[idx_1, : 2], dtype=tf.int32).numpy() 
#                 pt_2 = tf.cast(kps_out[idx_2, : 2], dtype=tf.int32).numpy()
#                 cv2.line(img_out_np, pt_1, pt_2, (255, 255, 255), line_width//2) 
#         return img_out_np

# ############################################################################
# #
# # 
# ############################################################################
# def topedown_pose_eval_img(upload,
#                         InferDet : callable,
#                         InferPose : callable):
    
#     def convert_images_into_np_array(img_jpg):
#         images_list = []
#         for img_name in img_jpg :
#             image = cv2.imread(img_name)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             image = np.array(image).reshape( ( image.shape[0], image.shape[1], 3)).astype(np.uint8)
#             images_list.append(image.copy())   
#             print(image.shape)
#             del image
#         return images_list
    


#     'convert image to np array for image plotting by cv2 lib'
#     image_np_list = convert_images_into_np_array(upload)

#     PoseDraw = coco_17kps_PoseDraw(num_joints = 17,
#                                 hm_thr = tf.ones(shape=(17,), dtype=tf.float32)*0.5,  
#                                 img_type = "only_left_parts")

#     for src_image_np in image_np_list :


#         src_image = tf.constant(src_image_np, dtype=tf.uint8)
#         src_image =  tf.expand_dims(src_image, axis=0)

#         'Inference of Single Person detection --------------------------------START'
#         bbox_yx_SrcImg = InferDet(src_image)

#         kps_pred_SrcImg = InferPose(src_image, bbox_yx_SrcImg)
#         'Inference of  TopDown Pose Detection-------------------------------- END'

#         'set circle_radius / line_width'
#         circle_radius = np.max((src_image_np.shape[0],src_image_np.shape[1]))
#         circle_radius = np.max( (circle_radius//100, 10))
#         line_width =  (circle_radius//2)
#         'show img with predicted bbox'
        
#         PoseDraw.drawing_with_kps_skeletons(kps_pred_SrcImg[0,:,:], 
#                                             bbox_yx_SrcImg[0,:],
#                                             src_image_np,
#                                             circle_radius = circle_radius,
#                                             line_width = line_width) 
                                     
#         plt.figure(figsize=(10,10))
#         plt.title(f'image with personal bbox @ src_img frame')
#         plt.imshow(src_image_np)
#         plt.show()

#         #im = Image.fromarray(img_out)
#         #im.save("filename.jpeg")
#         del src_image_np  
#         #break
#     return 0  