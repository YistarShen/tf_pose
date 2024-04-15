from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import sys, os
import numpy as np
import cv2
from matplotlib import rc, rcParams
rc('animation', html='jshtml')
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.animation import FuncAnimation, writers
#from mpl_toolkits.mplot3d import Axes3D
if os.path.basename(__file__) in os.listdir(os.getcwd()) :
    sys.path.append("../..")

#from lib.utils.utils import is_path_avaiable
from ipywidgets import IntProgress
from IPython.display import display
from time import time
import ipywidgets
''
from lib.utils.common import is_path_available




############################################################################
#
# 
############################################################################
def fig_2d_set(fig, subplot=133, tilte="Input"): 
    ax_in = fig.add_subplot(subplot)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title(tilte, fontsize=20)
    return ax_in

############################################################################
#
# 
############################################################################
def fig_3d_set(fig, RADIUS, subplot=131, azim=45, tilte="3D_GT"):  
    '3D Pose True'
    ax_true = fig.add_subplot(subplot, projection='3d')
    ax_true.set_title(tilte, fontsize=20)  # , pad=35
    ax_true.view_init(elev=15., azim=azim)
    ax_true.set_xlim3d([-RADIUS/2, RADIUS/2])
    ax_true.set_zlim3d([0, RADIUS])
    ax_true.set_ylim3d([-RADIUS/2, RADIUS/2])
    ax_true.set_xlabel("x")
    ax_true.set_ylabel("y")
    ax_true.set_zlabel("z")
    
    try:
        ax_true.set_aspect('equal')
    except NotImplementedError:
        ax_true.set_aspect('auto')

    return ax_true
############################################################################
#
# 
############################################################################
def cv2_decode_video(video_path,
                 poses_2D_data_len, 
                 cv2_rot_k = 0,
                 skip = 0):

    cap = cv2.VideoCapture(video_path)
    'video info'
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    effective_length = min(poses_2D_data_len,length)
    all_frames = []
    frame_counts = 0
    while(cap.isOpened()):

        was_read, frame = cap.read()
        if not was_read :
            print("test")
            break

        frame_counts = frame_counts+1

        if(frame_counts<skip):
            continue 

        'h36m videos property setting'
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if cv2_rot_k != 0 :
            frame = tf.image.rot90(frame, k=cv2_rot_k).numpy()  #(1920,1080,3)

        all_frames.append(frame)

        if frame_counts==effective_length:
            break

    return all_frames
############################################################################
#
# 
############################################################################
def save_animation(anim, output_file='test.mp4', fps=20):
    if output_file.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=3000)
        anim.save(output_file, writer=writer)
    elif output_file.endswith('.gif'):
        anim.save(output_file, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')

    return print(f"save file done: {output_file} @fps={fps}")


############################################################################
#
# 
############################################################################
def cv2_decode_h36m_video(video_path,
                poses_2D_data_len, 
                skip = 0):

    cap = cv2.VideoCapture(video_path)
    'video info'
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    effective_length = min(poses_2D_data_len,length)
    all_frames = []
    frame_counts = 0
    while(cap.isOpened()):

        was_read, frame = cap.read()
        if not was_read :
            print("test")
            break

        frame_counts = frame_counts+1

        if(frame_counts<skip):
            continue 
        
        'h36m videos property setting'
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame)

        if frame_counts==effective_length+skip:
            break

    return all_frames




############################################################################
#vis_h36m3d_HPE_resuts()
# 
############################################################################
def RenderAnimation3D_h36mStyle(
                        skip_frame : int = 0,        
                        poses_2D_true = None,
                        poses_2D_pred = None,
                        poses_3D_true = None,
                        poses_3D_pred = None,
                        skeltons = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15], 
                        joints_left = [4, 5, 6, 11, 12, 13],
                        viewport = (1000 , 1000),
                        azim=45,
                        save_video : bool = True,
                        save_dir='./',
                        extra_save_name = "2d_HPE_test.mp4",
                        input_video_path : Optional[str] = None,
                        joints_right_2d = [1, 2, 3, 14, 15, 16],
                        limit = None, 
                        figsize = (24,8),
                        fps=20):
    
    'video save name and path'
    if save_video :
        is_path_available(save_dir)
        video_name = os.path.basename(input_video_path)[:-4]
        video_save_path = os.path.join(save_dir, f'{video_name}_{fps}_{extra_save_name}.mp4')
        #video_save_path = os.path.join(save_dir, f'{extra_save_name}.mp4')
    else:
        video_name = 'test'
    '---------------info-----------------------------------------------'
    poses_2D_data_len, num_joints = poses_2D_pred.shape[0:2]
    if limit is None :
        limit = poses_2D_data_len
    '------------verify skeltons for plotting--------------------------------------------'
    assert len(skeltons) == num_joints , "num_joints of prediction doesn't match skeltons setting"

    '--------------------init------------------------------------------'
    lcolor="#3498db"
    rcolor="#e74c3c"
    interval = 1000/fps
    RADIUS = 1.7

    initialized = False
    lines_2D_true = []
    lines_2D_pred = []
    lines_3D_true = []
    lines_3D_pred = []
    points_2D = None
    points_3D = None
    image_true = None
    image_pred = None

    'Set plotting formart for evaluation mode with ground true or Demo Mode with only prediction'
    show_gt = True  if (poses_2D_true is not None) and (poses_3D_true is not None) else False


    fig = plt.figure(figsize=figsize)
    plt.suptitle(f'Video : {video_name} \n ( res_hw : {int(viewport[1])}x{int(viewport[0])}, fps : {int(fps)}, length : {limit})', fontsize=20)
    if show_gt :
        subplot_2d_true = 141
        subplot_2d_pred = 142
        subplot_3d_true = 143
        subplot_3d_pred = 144
    else:
        subplot_2d_pred = 121
        subplot_3d_pred = 122
    '------------------2D & 3D FIG--------------------------------------------'
    if show_gt :
        '2D Pose True'
        ax_2d_true = fig_2d_set(fig, subplot=subplot_2d_true, tilte="2D_kps_true")
        '3D Pose True'
        ax_3d_true = fig_3d_set(fig, RADIUS, subplot=subplot_3d_true, azim=azim, tilte="3D_GT")

    '2D Pose Pred'
    ax_2d_pred = fig_2d_set(fig, subplot=subplot_2d_pred, tilte="2D_kps_pred")
    '3D Pose Pred'
    ax_3d_pred = fig_3d_set(fig, RADIUS, subplot=subplot_3d_pred, azim=azim, tilte="3D_Model_Pred")

    if input_video_path is None:
        # Black background
        all_frames = np.zeros((poses_2D_pred.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        all_frames = cv2_decode_h36m_video(input_video_path, poses_2D_data_len, skip=skip_frame)


    '------------------init progress bar------------------------------'
    print(f"Total_Processing_Frames : {limit}")
    bar = IntProgress(layout = ipywidgets.Layout(width='auto', height='40px') )
    bar.max = limit
    bar.description = '(Init)'
    display(bar)
    

    '--------------------animation update function-------------------------------'
    def update_video(i):
        nonlocal initialized, image_true, image_pred, lines_2D_true, lines_2D_pred, points_2D, points_3D
        'update progress bar'
        bar.value = (i+1)
        #time.sleep(0.002)
        bar.description = f'frame:{i+1}'
        'true'
        if show_gt :
            trajectories_true = poses_3D_true[i,0,:] #(3,)
            ax_3d_true.set_xlim3d([-RADIUS/2 + trajectories_true[0], RADIUS/2 + trajectories_true[0]])
            ax_3d_true.set_ylim3d([-RADIUS/2 + trajectories_true[1], RADIUS/2 + trajectories_true[1]])
        'pred'
        trajectories_pred = poses_3D_pred[i,0,:] #(3,)
        ax_3d_pred.set_xlim3d([-RADIUS/2 + trajectories_pred[0], RADIUS/2 + trajectories_pred[0]])
        ax_3d_pred.set_ylim3d([-RADIUS/2 + trajectories_pred[1], RADIUS/2 + trajectories_pred[1]]) 
       
        'Update 2D poses color of skeltons'
        colors_2d = np.full(num_joints, 'black')
        colors_2d[joints_right_2d] = 'red'

        mask_joints = tf.math.count_nonzero(poses_3D_pred[i,:,:3], axis=-1)  #(17,3)=>(17,)
        mask_joints = tf.where(tf.equal(mask_joints,0)).numpy()
        eff_joint3d = tf.where(tf.not_equal(tf.math.count_nonzero(poses_3D_pred[i,:,:3], axis=-1)  ,0))
        eff_joint3d = tf.squeeze(eff_joint3d).numpy().tolist() 
    
        if not initialized :
            
            if show_gt :
                '2D_image'
                image_true = ax_2d_true.imshow(all_frames[0], aspect='equal') 
                '3D_image'
                init_data_true = poses_3D_true[0,...] #(17,3)
            '2D_image'
            image_pred = ax_2d_pred.imshow(all_frames[0], aspect='equal') 
            '3D_image'
            init_data_pred = poses_3D_pred[0,...] #(17,3)

            mask_joints = tf.math.count_nonzero(init_data_pred, axis=-1)  #(17,3)=>(17,)
            mask_joints = tf.where(tf.equal(mask_joints,0)).numpy()

            for j, j_parent in enumerate(skeltons):
                
                if j_parent == -1:
                    continue

                if j in mask_joints or j_parent in mask_joints :
                    continue

                if show_gt :
                    'Image with 2D Pose True'
                    lines_2D_true.append(ax_2d_true.plot([poses_2D_true[i, j, 0], poses_2D_true[i, j_parent, 0]],
                                        [poses_2D_true[i, j, 1], poses_2D_true[i, j_parent, 1]], color='pink')) 
                    'lines_3D_true'
                    lines_3D_true.append(ax_3d_true.plot([init_data_true[j,0], init_data_true[j_parent,0]], 
                                    [init_data_true[j,1], init_data_true[j_parent,1]], 
                                    [init_data_true[j,2], init_data_true[j_parent,2]],
                                    lw=2, c=lcolor if j in joints_left else rcolor))    
                'Image with 2D Pose Pred'     
                lines_2D_pred.append(ax_2d_pred.plot([poses_2D_pred[i, j, 0], poses_2D_pred[i, j_parent, 0]],
                                    [poses_2D_pred[i, j, 1], poses_2D_pred[i, j_parent, 1]], color='pink'))                    
                'lines_3D_Pred'
                lines_3D_pred.append(ax_3d_pred.plot([init_data_pred[j,0], init_data_pred[j_parent,0]], 
                                [init_data_pred[j,1], init_data_pred[j_parent,1]], 
                                [init_data_pred[j,2], init_data_pred[j_parent,2]],
                                lw=2, c=lcolor if j in joints_left else rcolor))
                
            points_2D = ax_2d_pred.scatter(*poses_2D_pred[i,:,:2].T, 30, color=colors_2d, edgecolors='white', zorder=10) 
            points_3D = ax_3d_pred.scatter(*init_data_pred.T, s=30, color='blue', edgecolors='white') 
            initialized = True

        else:
            if show_gt :
                '1. 2D image set data true '
                image_true.set_data(all_frames[i])
                pose_2D_in_true = poses_2D_true[i,:,:]
                '2. 3D image set data true '
                pose_3d_true = poses_3D_true[i,:,:]
            '3. 2D image set data pred'    
            image_pred.set_data(all_frames[i])
            pose_2D_in_pred = poses_2D_pred[i,:,:]
            '4. 3D image set data pred '
            pose_3d_pred = poses_3D_pred[i,:,:]
            
            for j, j_parent in enumerate(skeltons):
                if j_parent == -1:
                    continue
                if show_gt :
                    '2d true'
                    lines_2D_true[j-1][0].set_data([pose_2D_in_true[j, 0], pose_2D_in_true[j_parent, 0]],
                                        [pose_2D_in_true[j, 1], pose_2D_in_true[j_parent, 1]])
                    '3d true'
                    lines_3D_true[j-1][0].set_xdata(np.array([pose_3d_true[j, 0], pose_3d_true[j_parent, 0]]))
                    lines_3D_true[j-1][0].set_ydata(np.array([pose_3d_true[j, 1], pose_3d_true[j_parent, 1]]))
                    lines_3D_true[j-1][0].set_3d_properties(np.array([pose_3d_true[j, 2], pose_3d_true[j_parent, 2]]), zdir='z')
                '2d pred'
                lines_2D_pred[j-1][0].set_data([pose_2D_in_pred[j, 0], pose_2D_in_pred[j_parent, 0]],
                                    [pose_2D_in_pred[j, 1], pose_2D_in_pred[j_parent, 1]])
                '3d pred'
                lines_3D_pred[j-1][0].set_xdata(np.array([pose_3d_pred[j, 0], pose_3d_pred[j_parent, 0]]))
                lines_3D_pred[j-1][0].set_ydata(np.array([pose_3d_pred[j, 1], pose_3d_pred[j_parent, 1]]))
                lines_3D_pred[j-1][0].set_3d_properties(np.array([pose_3d_pred[j, 2], pose_3d_pred[j_parent, 2]]), zdir='z')   

                'Plot 2D keypoints'
                #points_2D.set_offsets(poses_2D[i])  
                # 
            'Plot 2D/3D key points ' 
            points_2D.set_offsets(pose_2D_in_pred)   
            points_3D._offsets3d = (pose_3d_pred[ eff_joint3d,0], pose_3d_pred[eff_joint3d,1], pose_3d_pred[eff_joint3d,2])             
        #fig.tight_layout()
        plt.show()
    
    anim = FuncAnimation(fig, 
                update_video, 
                frames=np.arange(0, limit), 
                interval=interval, 
                blit=False, 
                repeat=True) 
    
    if save_video :
        save_animation(anim, output_file=video_save_path, fps=fps)


    '=============================bar final update============================='
    bar.description = f'Done ({limit})'
    return anim