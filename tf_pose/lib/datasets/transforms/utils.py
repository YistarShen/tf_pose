
import copy, warnings
import tensorflow as tf
from tensorflow import Tensor
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from lib.Registers import DATASETS

#----------------------------------------------------------------
#
#-----------------------------------------------------------------
def ReviewPoseTargets(reviews : int = 5,
                    batch_size : int =1, 
                    shuffle : bool= False,
                    tfrec_datasets_list : List[Dict] = [], 
                    transforms : List[Union[Dict, Callable]] = [], 
                    codec : Optional[Union[Dict,Callable]]=None, 
                    img_inverse=False,
                    show_kps_log = False):
    
    def tf_img_norm_transform(img, inv=False):
        img_mean = tf.constant([0.485, 0.456, 0.406],dtype=tf.float32)
        img_std = tf.constant([0.229, 0.224, 0.225],dtype=tf.float32)
        if (inv==False):
            img = img / 255.0
            img = (img - img_mean)/img_std
        else:
            img =  img*img_std + img_mean
        return img

    assert batch_size>0 and isinstance(batch_size, int), \
    "batch_size must be >=1 and int type" 


    test_dataloader_cfg =  dict(
        type = 'dataloader',
        batch_size = batch_size,
        prefetch_size = 4,
        shuffle  =  shuffle,
        tfrec_datasets_list = tfrec_datasets_list,
        augmenters = transforms,
        codec = codec
    )

    tfds_builder = DATASETS.build(copy.deepcopy(test_dataloader_cfg))
    batch_dataset = tfds_builder.GenerateTargets(test_mode=False) 

    for samples in batch_dataset.take(reviews):

        batch_images, batch_heatmaps, batch_kps = samples

        if img_inverse :
            batch_images = tf_img_norm_transform(batch_images, inv=img_inverse)

        print("heatmaps.shape : ",batch_heatmaps.shape)
        print("kps.shape : ",batch_kps.shape)
        if show_kps_log :
            print("kps : ",batch_kps[0,...])

  
        if hasattr(tfds_builder, 'codec') :
            batch_kps_test, _ = tfds_builder.codec.batch_decode(batch_heatmaps)

        plt.figure(figsize=(15,25))
        'crop image' 
        plt.subplot(1,3,1)
        plt.title('crop image with kps',fontsize= 20)
        for i in range(0,batch_kps.shape[1]):
            kps_x = int((batch_kps[0,i,0]))
            kps_y = int((batch_kps[0,i,1]))
            plt.scatter(kps_x,kps_y)
        plt.imshow(batch_images[0,:,:,:])  

        plt.subplot(1,3,2)
        plt.title('crop image with kps from heatmaps',fontsize= 20)
        for i in range(0,batch_kps_test.shape[1]):
            kps_x = int((batch_kps_test[0,i,0]))
            kps_y = int((batch_kps_test[0,i,1]))
            plt.scatter(kps_x,kps_y)
        plt.imshow(batch_images[0,:,:,:])        

        'heatmaps'
        plt.subplot(1,3,3)
        plt.title('kps heatmap sum',fontsize= 20)
        heatmaps_plot = np.array(batch_heatmaps[0,:, :, :])
        plt.imshow(heatmaps_plot.sum(axis=2)) 



#----------------------------------------------------------------
#
#-----------------------------------------------------------------
def cv_transform_pose_test(reviews : int = 5,
                        batch_size : int =1, 
                        shuffle : bool= False,
                        tfrec_datasets_list : List[Dict] = [], 
                        transforms : List[Union[Dict, Callable]] = [], 
                        codec : Optional[Union[Dict,Callable]]=None,
                        show_transform2src : bool = False):
    '''
    assert isinstance(tfrec_datasets, list)
    for tfrec_dataset in tfrec_datasets:
        assert isinstance(tfrec_dataset, dict)
    '''
    assert batch_size>0 and isinstance(batch_size, int), \
    "batch_size must be >=1 and int type"

    test_dataloader_cfg =  dict(
        type = 'dataloader',
        batch_size = batch_size,
        prefetch_size = 4,
        shuffle  =  shuffle,
        tfrec_datasets_list = tfrec_datasets_list,
        augmenters = transforms,
        codec = codec
    )



    tfds_builder = DATASETS.build(copy.deepcopy(test_dataloader_cfg))
    batch_dataset = tfds_builder.GenerateTargets(test_mode=True)

    '----------------sample keys----------------------------------------'
    samples = next(iter(batch_dataset))
    # #print(samples.keys())
    assert isinstance(samples,dict), "samples must be dict type @cv_transform_test"
    for key in samples.keys() :
        if isinstance(samples[key], (tf.Tensor,tf.RaggedTensor)):
            print(f"{key}: shape - {samples[key].shape}")
        else:
            assert isinstance(samples[key], dict)
            print(f"{key}: ")
            for sub_key in samples[key].keys():
                #print(f"{sub_key}: sub_dict - [{features[key][sub_key]}]") 
                shape = samples[key][sub_key].shape
                print(f"      {sub_key}: {shape if shape != () else samples[key][sub_key]}")
    
    sub_plots = 2
    is_resized_img = False
    for samples in batch_dataset.take(reviews):
        #assert isinstance(samples,dict), "samples must be dict type @cv_transform_test"
        #print(samples.keys())
        
        ' transformed data'
        batch_images = samples['image']
        batch_kps_true = samples['kps']
        batch_bbox = samples['bbox']
        batch_image_size = samples['image_size']

        ' meta_info of source image '
        batch_src_images = samples['meta_info']['src_image']
        batch_src_kps_true = samples['meta_info']['src_keypoints']
        batch_src_bbox = samples['meta_info']['src_bbox']
        batch_instance_id = tf.get_static_value(samples['meta_info']['id'])[0]
        batch_image_id = tf.get_static_value(samples['meta_info']['image_id'])[0]
        batch_src_kps_true = tf.reshape(batch_src_kps_true,(-1,batch_kps_true.shape[1],3))
        
        print("batch_images spec : ", tf.type_spec_from_value(batch_images))
        print("kps_true.shape : ",batch_kps_true.shape)
        print("batch_src_images spec : ",tf.type_spec_from_value(batch_src_images))
        print("batch_src_kps_true.shape : ",batch_src_kps_true.shape)

        if batch_images[0,...].shape!=batch_src_images[0,...].shape:
            is_resized_img = True
    
        ####################################################################################
        'print info for id = 0 in batch'
        print(f"\n\n ------------< img_id :{batch_image_id},  id:{batch_instance_id} >--------------------")
        print(f"    image_size : {batch_image_size[0,:]} @ is_resize={is_resized_img}")
        print(f"    bbox : {batch_bbox[0,:]}")
        tf.print(f"    kps : {batch_kps_true[0,...]}")

        if samples.get('bbox_center', None) is not None:
            print(f"    bbox_center : {samples['bbox_center'][0,:]}")
        if samples.get('bbox_scale', None) is not None:
            print(f"    bbox_scale : {samples['bbox_scale'][0,:]}")   

        if show_transform2src:
            transform2src = samples['transform2src']
            print(f"    scale_xy : {transform2src['scale_xy'][0,:]}")
            print(f"    pad_offset_xy : {transform2src['pad_offset_xy'][0,:]}")
            print(f"    bbox_lt_xy : {transform2src['bbox_lt_xy'][0,:]}")

        ####################################################################################
        
        if samples.get('y_true', None) is not None:
            batch_heatmaps = samples['y_true'] 
            if hasattr(tfds_builder, 'codec') :
                batch_kps_test, _ = tfds_builder.codec.batch_decode(batch_heatmaps)
                sub_plots = 4
            else:
                sub_plots = 3
                warnings.warn("tfds_builder has no attribute 'codec' " ) 
                
        if samples.get('sample_weight', None) is not None:
            sample_weight = samples['sample_weight']    
        
        sub_plot_id = [i+1 for i in range(sub_plots)]
        ####################################################################################


        if isinstance(batch_src_images, tf.Tensor) :
            batch_src_image = batch_src_images[0,:,:,:]
        else:
            batch_src_image = batch_src_images[0,:,:,:].to_tensor()

        if isinstance(batch_images, tf.Tensor) :
            batch_image = batch_images[0,:,:,:]
        else:
            batch_image = batch_images[0,:,:,:].to_tensor()
            
         ####################################################################################

        plt.figure(figsize=(40,15))
        plt.subplot(1,sub_plots,sub_plot_id[0])
        plt.title(f'src_img with kps_true', fontsize= 24)
        for i in range(0,batch_src_kps_true.shape[1]):
            kps_x = int((batch_src_kps_true[0,i,0]))
            kps_y = int((batch_src_kps_true[0,i,1]))
            plt.scatter(kps_x,kps_y)
        plt.imshow(batch_src_image)
        
        print( batch_src_bbox[0,:])
        x1, y1, w, h = batch_src_bbox[0,:]
        ax = plt.gca()
        patch = plt.Rectangle([x1, y1], w, h, fill=False, edgecolor=[1, 0, 1], linewidth=5)
        ax.add_patch(patch)
    

        'crop/resized image' 
        plt.subplot(1,sub_plots,sub_plot_id[1])
        plt.title('resized img with kps_true',fontsize= 24)
        for i in range(0,batch_kps_true.shape[1]):
            kps_x = int((batch_kps_true[0,i,0]))
            kps_y = int((batch_kps_true[0,i,1]))
            plt.scatter(kps_x,kps_y)
        plt.imshow(batch_image)
        if codec is None :
            x1, y1, w, h = batch_bbox[0,:]
            ax = plt.gca()
            patch = plt.Rectangle([x1, y1], w, h, fill=False, edgecolor=[1, 0, 1], linewidth=5)
            ax.add_patch(patch)

        if sub_plots==4 :
            plt.subplot(1,sub_plots,sub_plot_id[2])
            plt.title('kps from hm', fontsize= 24)
            for i in range(0,batch_kps_test.shape[1]):
                kps_x = int((batch_kps_test[0,i,0]))
                kps_y = int((batch_kps_test[0,i,1]))
                plt.scatter(kps_x,kps_y)
            plt.imshow(batch_image)

        if  sub_plots==3 or sub_plots==4 :
            plt.subplot(1,sub_plots,sub_plot_id[-1]) 
            plt.title('heatmap',fontsize= 24)
            heatmaps_plot = np.array(batch_heatmaps[0,:, :, :])
            plt.imshow(heatmaps_plot.sum(axis=-1))
            #print(heatmaps_plot.sum(axis=-1).shape)
        
        plt.suptitle(f'< img_id :{batch_image_id},  id:{batch_instance_id} >',fontsize= 36)
        #plt.tight_layout()
        plt.show()

    del batch_dataset
    #return tfds_builder


#----------------------------------------------------------------
#
#-----------------------------------------------------------------
def PackInputTensorTypeSpec(data : Tuple[Union[Tuple, Dict, tf.Tensor,tf.RaggedTensor]], 
                            TensorSpec = {},
                            show_log : bool = True) ->Dict[str,tf.TensorSpec]:

    assert isinstance(data, (Tuple, Dict, tf.Tensor,tf.RaggedTensor)) or data==None, \
    f"input data must be dict or Tuple type, but got {type(data)} @tensor_type_spec"
    

    if isinstance(data, (tf.Tensor,tf.RaggedTensor)): 
       TensorSpec += (tf.type_spec_from_value(data),)
       return TensorSpec
    if isinstance(data, Dict):
        values = data.values()
        keys = data.keys()
    elif isinstance(data, Tuple):
        assert len(data)<=3, f"if data is tuple type, len(data) must be <=3, but got {len(data)} "
        values = data
        TensorSpec_Parser_list = ['image','y_true','sample_weight']
        keys = TensorSpec_Parser_list[:len(data)]

    else:
        raise TypeError(
           "valid data type must be Dict, Tuple , tf.Tensor or tf.RaggedTensor"
           f", but got type(data) @PackInputTensorTypeSpec"
        )   

    for key, val in zip(keys, values):   
        if isinstance(val, dict):
            sub_TensorSpec = PackInputTensorTypeSpec(val,{},show_log)
            # if isinstance(sub_TensorSpec, Dict):
            #     TensorSpec[key] = sub_TensorSpec
        elif  isinstance(val, tuple):
            sub_TensorSpec = tuple()
            for sub_val in val :
                if  not isinstance(sub_val, (tf.Tensor,tf.RaggedTensor)):
                    raise TypeError(
                            "value type must be tf.Tensor ot tf.RaggedTensor"
                    )
                #sub_TensorSpec += (tf.type_spec_from_value(sub_val),)
                sub_TensorSpec += PackInputTensorTypeSpec(sub_val, (), show_log)
            TensorSpec[key] = sub_TensorSpec
        elif isinstance(val, (tf.Tensor,tf.RaggedTensor)):
            TensorSpec[key] = tf.type_spec_from_value(val)
        elif val == None :
            TensorSpec[key] = None
        else:
            raise TypeError(
                "value type must be tf.Tensor ot tf.RaggedTensor"
                )
        
    if show_log:
        if isinstance(data, dict):
            print("\n==========Pack data type : Dict[str,Tensor] --Test Mode ============= ")
        if isinstance(data, Tuple):
            print(f"\n==========Pack data type : Tuple[Tensor] --- <<{(TensorSpec_Parser_list)}>> ============= ")
        if isinstance(data, (tf.Tensor,tf.RaggedTensor)):
            print("\n==========Pack data type : single tensor ============= ")
        for key in TensorSpec.keys():
            print(key, "-----", TensorSpec[key])
    return TensorSpec


#----------------------------------------------------------------
#
#-----------------------------------------------------------------
def rand_prob(shape=(),  seed=None):
    return tf.random.uniform(shape=shape, 
                             minval=0.,
                             maxval=1., 
                             seed=seed)

#----------------------------------------------------------------
#
#-----------------------------------------------------------------
def rand_inverse(shape=(), seed=None):
    prob = tf.random.uniform(shape=shape, minval=0.,maxval=1.,seed=seed)
    return tf.cond(prob>0.5, 
                   lambda : 1., 
                   lambda: -1.)

#----------------------------------------------------------------
#
#-----------------------------------------------------------------
def rand_bool(shape=(), seed=None):
    prob = tf.random.uniform(shape=shape, 
                             minval=0.,
                             maxval=1.,
                             seed=seed)
    return tf.cond(prob>0.5, 
                   lambda : True, 
                   lambda : False)

#----------------------------------------------------------------
#
#-----------------------------------------------------------------
def rand_value(shape=(),  minval=-1., maxval=1., seed=None):
    return tf.random.uniform(shape=shape, 
                             minval=minval,
                             maxval=maxval, 
                             seed=seed)


