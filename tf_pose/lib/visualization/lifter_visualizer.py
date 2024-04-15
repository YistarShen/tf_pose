import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
rc('animation', html='jshtml')
import numpy as np
#--------------------------------------------------------------------------------------------
#
#
#--------------------------------------------------------------------------------------------
def render_animation(poses_3D_data, 
            skeltons, 
            joints_left,
            mask_joints = [],
            root_id = 0,
            frames=1000, 
            fps=20, 
            limit=-1):

  #l_parts = [-1,1,1,1,0,0,0,1,1,1,1,0,0,0,1,1,1]
  lcolor="#3498db"
  rcolor="#e74c3c"
  interval = 1000/fps
  RADIUS = 1.7
  'init'
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_title("3D_Test")  # , pad=35
  ax.view_init(elev=15., azim=45)
  ax.set_xlim3d([-RADIUS/2, RADIUS/2])
  ax.set_zlim3d([0, RADIUS])
  ax.set_ylim3d([-RADIUS/2, RADIUS/2])
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  try:
    ax.set_aspect('equal')
  except NotImplementedError:
    ax.set_aspect('auto')
  #ax.dist = 10
  initialized = False
  lines_3D = []

  points_3D = None
  points_3D_test = []

  'Update 2D poses color of skeltons'
  colors_3d = np.full(17, 'black')
  colors_3d[root_id] = 'red' 

  def update_video(i):
    nonlocal initialized, lines_3D, points_3D, points_3D_test
    'test'
    #poses_3D_data[i,0,:] = 0 #(3,)
    trajectories = poses_3D_data[i,0,:] #(3,)
    ax.set_xlim3d([-RADIUS/2 + trajectories[0], RADIUS/2 + trajectories[0]])
    ax.set_ylim3d([-RADIUS/2 + trajectories[1], RADIUS/2 + trajectories[1]])

    mask_3d_lines = len(mask_joints)
    
    if not initialized :
      init_data = poses_3D_data[0,...] #(17,3)
     
      for j, j_parent in enumerate(skeltons):
      
        if j_parent == -1:
          continue
        
        if j in mask_joints or j_parent in mask_joints :
          continue

        lines_3D.append(ax.plot([init_data[j,0], init_data[j_parent,0]], 
                    [init_data[j,1], init_data[j_parent,1]], 
                    [init_data[j,2], init_data[j_parent,2]],
                    lw=2, c=lcolor if j in joints_left else rcolor ))
        '''
        points_3D_test.append(ax.scatter(init_data[j,0], 
                          init_data[j,1], 
                          init_data[j,2],
                          s=30,  
                          edgecolors='white'))
        '''
        #points_3D_test.append(ax.scatter(*init_data[j,:], s=30, color='black', edgecolors='white')) 
      
      #print(len(points_3D_test))
            
      #points_3D_test.append(ax.scatter(*init_data[root_id,:], s=50, color='red', edgecolors='white')) 
      points_3D = ax.scatter(init_data[:,0], init_data[:,1], init_data[:,2], s=30, color=colors_3d, edgecolors='white')

      initialized = True
    else:
      pose = poses_3D_data[i,:,:]
      for j, j_parent in enumerate(skeltons):

        if j_parent == -1:
          continue
        
        if j in mask_joints or j_parent in mask_joints :
          continue

        lines_3D[j-1-mask_3d_lines][0].set_xdata(np.array([pose[j, 0], pose[j_parent, 0]]))
        lines_3D[j-1-mask_3d_lines][0].set_ydata(np.array([pose[j, 1], pose[j_parent, 1]]))
        lines_3D[j-1-mask_3d_lines][0].set_3d_properties(np.array([pose[j, 2], pose[j_parent, 2]]), zdir='z')
        #print(j-1-mask_3d_lines,j, points_3D_test[8])
        #points_3D_test[j-1-mask_3d_lines]._offsets3d = (pose[j,0], pose[j,1], pose[j,2]) 
        #if j < 15 :
          #points_3D_test[j-1-mask_3d_lines]._offsets3d = (10, 10, 10) 
        #print("test")

      points_3D._offsets3d = (pose[:,0], pose[:,1], pose[:,2])
      #points_3D = ax.scatter(pose[:,0], pose[:,1], pose[:,2])
      #points_3D_test[-1]._offsets3d = (pose[root_id,0], pose[root_id,1], pose[root_id,2])


  anim = FuncAnimation(fig, update_video, 
              frames=frames, interval=interval, blit=False, repeat=True) 

  return anim



#--------------------------------------------------------------------------------------------
#
#
#--------------------------------------------------------------------------------------------
def render_animation_2D(keypoints, 
             skeltons, 
             joints_left,
             viewport, 
             frames=1000, fps=20, limit=-1):
  
  interval = 1000/fps
  initialized = False
  #plt.ioff()
  plt.ioff()
  #plt.ion()
  fig = plt.figure()
  ax_in = fig.add_subplot(111)
  ax_in.get_xaxis().set_visible(False)
  ax_in.get_yaxis().set_visible(False)
  ax_in.set_axis_off()
  ax_in.set_title('Input')


  # Update 2D poses
  #joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
  #colors_2d = np.full(keypoints.shape[1], 'black')
  #colors_2d[joints_right_2d] = 'red'
  lines = []
  points = None
  image = None
  #print(len(skeltons), keypoints.shape[1])

  input_video_path = None

  # Decode video
  if input_video_path is None:
    # Black background
    all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
  else:
    print("test")


  def update_video_2D(i):
    nonlocal initialized, image, lines, points

    if not initialized:
      for j, j_parent in enumerate(skeltons):
        image = ax_in.imshow(all_frames[i], aspect='equal')
        if j_parent == -1:
          continue
        #if len(skeltons) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
        if len(skeltons) == keypoints.shape[1]:
          # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
          lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                       [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))
          
      #points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)
      initialized = True
    else:
      image.set_data(all_frames[i])
      
      for j, j_parent in enumerate(skeltons):
        if j_parent == -1:
          continue
        if len(skeltons) == keypoints.shape[1]:
          lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                        [keypoints[i, j, 1], keypoints[i, j_parent, 1]]) 
      #points.set_offsets(keypoints[i])

  anim = FuncAnimation(fig, update_video_2D, 
              frames=frames, 
              interval=interval, 
              blit=False, repeat=True) 
  
  #plt.close()
  #plt.ioff()
  #plt.show()
  return anim  







