import tensorflow as tf
from tensorflow import Tensor
from typing import Dict, List, Optional, Tuple, Union
import tensorflow_addons as tfa
from lib.datasets.transforms.utils import rand_prob
from lib.Registers import TRANSFORMS
from lib.datasets.transforms import RandomPadImageResize
from .base import CVBaseTransformLayer
#---------------------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------------------
@TRANSFORMS.register_module()
class Mosaic(CVBaseTransformLayer):
    VERSION = '2.3.0'
    r""" Mosaic implements the mosaic data augmentation technique.
    date : 2024/3/25
    author : Dr. Shen 

    to do : 1. add bbox format converter , to ensure all type input can work well
            4. verify ImageProjectiveTransformV3
            6. verify input is tensor not ragged tensor



    Required Keys:
        - image ----- TensorSpec(shape=(b, h, w, 3), dtype=tf.uint8, name=None)
        - labels ----- RaggedTensorSpec(TensorShape([b, None]), tf.int32, 1, tf.int64) 
        - bbox ----- RaggedTensorSpec(TensorShape([b, None, 4]), tf.float32, 1, tf.int64)
        - kps(optional) ----- RaggedTensorSpec(TensorShape([b, None, num_joints, 3]), tf.float32, 1, tf.int64)

    Modified Keys:
        - image 
        - labels
        - bbox
        - kps(optional)



    Args:
        target_size (tuple[int]): output dimension after mosaic transform, `[height, width]`.
            If `None`, output is the same size as input image . defaults to (640,640).
        interpolation (str): Interpolation mode. Supported values: "nearest","bilinear". Defaults to 'bilinear'
        center_ratio_range (Tuple[float] ): 
        scaling_ratio_range (Tuple[float]) : Min and max ratio of scaling  used in affine transform. 
            if use_affine_transform = False, it is invalid, Defaults to (0.75, 1.25).
        num_used_kps (int) : number of joints used for each pose object, Defaults to 0
           for object detection training task,  num_used_kps = 0
           for bottomup pose detection training task,  num_used_kps = 17 if using standard coco style keypoints
        pose_obj_label_id (int) : the label index of pose object, Defaults to 1
        img_padding_val (int) : padding value for each image
        label_padding_val (int) : padding value for labels id corresponding to invalid bboxes 
            invalid bboxes are outside mosaic image size. Defaults to 0
        use_affine_transform (bool) : whether to use affine_transform to crop and resize image , Default to True
        min_bbox_area (float) : minimum area of bbox for labeling, if min_bbox_area!=0., 
            transform will filter out instances which of bboox area is greater than min_bbox_area. Defualt to 0
        parallel_iterations (int) : The number of iterations allowed to run in parallel for tf.map_fn. Default to 32    
        prob (float) : The probability to enable MixUp . Defaults to 0.5

        

    References:
        - [Yolov4  paper] (https://arxiv.org/pdf/2004.10934).
        - [Yolov5 implementation](https://github.com/ultralytics/yolov5).
        - [YoloX implementation](https://github.com/Megvii-BaseDetection/YOLOX)
        - ['Mosaic' implement of Yolov8 ] (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py).
        - ['Mosaic' implement of keras_cv] (https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing/mosaic.py).
        - ['Mosaic' implement of mmdet] ( https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/transforms/transforms.py).

    Sample usage:

    ```python
    import copy
    from lib.Registers import DATASETS
    from lib.datasets.transforms import Mosaic, MixUp
    ---------------------------------------------------------------  
    tfrec_val_dataset = dict(
        type = 'Test_CocoStylePoseDataset_Parser', 
        data_root = val_tfrec_bottmup_pose_dir,
        is_whole_body_boxes = False,
        sel_kpt_idx  = [i for i in range(17)],
    )

    mosaic = Mosaic( 
        target_size = (640,640),
        interpolation = 'bilinear',
        center_ratio_range = (0.75, 1.25),
        scaling_ratio_range= (0.75, 1.25),
        num_used_kps = 17,
        min_bbox_area = 0., #4096./2,
        use_affine_transform= True,
        prob = 1.
    )

    ---------------------------------------------------------------  
    test_dataloader_cfg =  dict(
        type = 'dataloader',
        batch_size = 16,
        prefetch_size = 4,
        shuffle  =  True,
        tfrec_datasets_list = [tfrec_val_dataset],
        augmenters = [mosaic], 
        codec = None
    )

    tfds_builder = DATASETS.build(
        copy.deepcopy(test_dataloader_cfg)
    )
    batch_dataset = tfds_builder.GenerateTargets(
        test_mode=False, 
        unpack_x_y_sample_weight= False, 
        ds_weights =None
    )
    --------------------------------------------------------------- 
    tfds_builder.get_pipeline_cfg() 
    batch_id = 0
    for features in batch_dataset.take(1):
        PackInputTensorTypeSpec(features,{}, show_log=True)
        image = features["image"][batch_id].numpy()
        labels = features['labels'][batch_id]
        bboxes = features['bbox'][batch_id]
        
        if features.get('kps',None) is not None :
            kpts= features['kps'][batch_id] 
            print(f" Shape = [labels : {labels.shape}, bboxes : {bboxes.shape}, kpts : {kpts.shape} ]")
            pose_obj_num = tf.reduce_sum(
                tf.cast(labels==1, dtype=tf.int32)
            )
            print(f"labels : {labels}, pose_obj_num : {pose_obj_num}") 
        
        plt.figure(figsize=(10, 10))
        if  features.get('meta_info',None) is not None :
            plt.title(
                f"image_id: {features['meta_info']['image_id'][batch_id]}"
            )
        plt.imshow(image)

        ith_obj = -1
        for i, (bbox,label) in enumerate(zip(bboxes,labels)):
            if label==1:
                color = np.random.uniform(
                    low=0., high=1., size=3
                )
                ith_obj +=1
            ax = plt.gca()
            x1, y1, w, h = bbox
            patch = plt.Rectangle(
                [x1, y1], w, h, fill=False, edgecolor=color, linewidth=2
            )
            ax.add_patch(patch)
            text = "{}".format(label)
            ax.text(
                x1, y1, text,
                bbox={"facecolor": color, "alpha": 0.8},
                clip_box=ax.clipbox,
                clip_on=True,
            )

            if features.get('kps',None) is None :
                continue
            kpt_ith = kpts[ith_obj]
            for j in range(0,kpt_ith.shape[0]):
                kps_x = int((kpt_ith[j,0]))
                kps_y = int((kpt_ith[j,1]))
                plt.scatter(kps_x,kps_y, color=color)
    self.compute_dtype
    """

    def __init__(
        self,
        target_size : Union[Tuple,List] = (640,640),
        interpolation : str = 'bilinear',
        center_ratio_range : Tuple[float] = (0.75, 1.25),
        scaling_ratio_range: Tuple[float, float] = (0.75, 1.25),
        num_used_kps : int = 0,
        pose_obj_label_id : int = 1,
        img_padding_val : int = 0,
        label_padding_val : int =  0,
        use_affine_transform : bool = False,
        min_bbox_area : Optional[float] = None,
        parallel_iterations : int = 32,
        prob: float = 0.,
        **kwargs
    ):
        super().__init__(**kwargs)
        'method'
        self.apply_affine_transform = use_affine_transform
        
        #bbox_padding_val
        'basic cfg'

        self.prob = prob
        self.pose_obj_label_id = pose_obj_label_id
        self.label_padding_val = label_padding_val
        self.fill_value = img_padding_val
        self.interpolation = interpolation
        self.target_size = target_size
        self.target_size_wh_tensor = tf.cast(
            [target_size[1],target_size[0]], dtype=self.compute_dtype
        )
        self.lt_ratio_range = (center_ratio_range[0]-0.5, center_ratio_range[1]-0.5)
        self.scaling_ratio_range = scaling_ratio_range
        self.min_bbox_area =  0. if min_bbox_area is None else min_bbox_area



        self.padding_resize = RandomPadImageResize(
            target_size = (640,640),
            pad_types  = ['center'],
            pad_val = 0,
            use_udp  = False,
            dtype= self.compute_dtype
        )


        'tf.map_fn'
        self.parallel_iterations = parallel_iterations
        self.num_used_kps = num_used_kps
        self.use_kps = num_used_kps>0

    def _combine_mosaic4(self, mosaic_data, batch_size):
        r''''
        - input mosaic_data
        mosaic_data['mosaic_images'] : (4,None,None,3) @RaggedTensor__uint8
        mosaic_data['mosaic_bboxes'] : (4,None,4) @RaggedTensor__flaot32
        mosaic_data['mosaic_labels'] : (4,None) @RaggedTensor__int32

        - output mosaic_data
        mosaic_data['mosaic_images'] : (640*2,640*2,3) @Tensorr__uint8
        mosaic_data['mosaic_bboxes'] : (None,4) @RaggedTensor__flaot32
        mosaic_data['mosaic_labels'] : (None,) @RaggedTensor__int32
        '''


        mosaic_data['image'] = mosaic_data['image'].merge_dims(0,1)
        mosaic_data['bbox'] = mosaic_data['bbox'].merge_dims(0,1)
        mosaic_data['labels'] = mosaic_data['labels'].merge_dims(0,1)
        if mosaic_data.get('kps',None)!=None :
            mosaic_data['kps'] = mosaic_data['kps'].merge_dims(0,1)

        'image'
        mosaic_data = self.padding_resize(
            mosaic_data,
            tf.tile( 
                tf.cast(['rb','lb','rt','lt'], dtype=tf.string ),[batch_size]
            )
        )
        #lt_imgs, rt_imgs,  lb_imgs, rb_imgs = tf.split( data['image'], 4)
        lt_imgs, rt_imgs,  lb_imgs, rb_imgs  = self.mosaic4x_data_split(
            mosaic_data['image']
        )
        #tf.print(lt_imgs.get_shape())
        mosaic_data['image'] = tf.concat(
            [
                tf.concat([lt_imgs, rt_imgs], axis=2),  # upper_stack
                tf.concat([lb_imgs, rb_imgs], axis=2)   # lower_stack
            ],
            axis=1
        )

        'bbox'
        lt_bbox, rt_bbox,  lb_bbox, rb_bbox = self.mosaic4x_data_split(mosaic_data['bbox'])
        mosaic_data['bbox'] = tf.concat(
            [
                lt_bbox,
                rt_bbox + tf.cast([self.target_size[1],0,0,0], dtype=self.compute_dtype),
                lb_bbox+ tf.cast([0,self.target_size[0],0,0], dtype=self.compute_dtype),
                rb_bbox+ tf.cast([self.target_size[1],self.target_size[0],0,0], dtype=self.compute_dtype)
            ],
            axis=1
        ) #(Ragged_num_obj,4) RaggedTensor

        'labels'
        lt_labels, rt_labels,  lb_labels, rb_labels= self.mosaic4x_data_split(
            mosaic_data['labels']
        )
        mosaic_data['labels'] = tf.concat(
            [lt_labels, rt_labels,lb_labels, rb_labels],
            axis=1
        ) #(Ragged_num_obj,4) RaggedTensor

        if mosaic_data.get('kps',None)!=None :
            lt_kps, rt_kps,  lb_kps, rb_kps = self.mosaic4x_data_split(
                mosaic_data['kps']
            )
            mosaic_data['kps']= tf.concat(
                [
                    lt_kps,
                    rt_kps + tf.cast([self.target_size[1],0, 0], dtype=self.compute_dtype),
                    lb_kps + tf.cast([0,self.target_size[0], 0], dtype=self.compute_dtype),
                    rb_kps+ tf.cast([self.target_size[1],self.target_size[0],0], dtype=self.compute_dtype)
                ],
                axis=1
            )#(None,17,3)
        return mosaic_data
            

    def _ragged_xy_points_affine_transform(self, pt_xy, matAffine_2x3):
        """ 
        pt_xy           pt_xy                   associated
        shape_rank      shape                   matAffine_2x3   
        --------------------------------------- -------------------------------
        3            (b, b1, b2, 2)             (b,None, None, 2,3)   
        2            (b, b1, 2,)                (b,None, 2,3)       
        1            (b,2)                      (b,2,3)  
        """
        # pt_shape_rank = pt_xy.shape.rank #(1,2,3)
        # for _ in range (pt_shape_rank-2):
        #     matAffine_2x3 = matAffine_2x3[:,None,...]
        matAffine_2x3 = matAffine_2x3[:,None,None,...]   #(b,None, None, 2,3)   
 
        pt_3x1 = tf.expand_dims( 
            tf.concat(
                [pt_xy[...,:2], tf.ones_like(pt_xy[...,:1])],
                axis=-1
            ) ,
            axis=-1
        ) #(points,3,1) 
        #return pt_xy
    
        new_pt = tf.map_fn(
            fn=lambda x: tf.matmul(x[0],x[1]), 
            elems=(matAffine_2x3,pt_3x1),
            fn_output_signature = tf.RaggedTensorSpec(
                tf.TensorShape([None,None,2,1]), ragged_rank=0, dtype=self.compute_dtype
            ) 
        ) #(b,num_obj,2,2,1) or (b,num_pose_obj,17,2,1)
        #new_pt = tf.linalg.matmul(matAffine_2x3,pt_3x1, b_is_sparse=True) #(17,2,1)  (None,2,3)@(points,3,1)=  (None,3,1)
        return tf.squeeze(new_pt, axis=-1)
    
    def affine_transform(self, data,  batch_size=None) :

        trans_xy = tf.random.uniform( 
            shape=(batch_size,2),
            minval=self.lt_ratio_range[0], #0.25
            maxval=self.lt_ratio_range[1], #0.75
            dtype=self.compute_dtype,
        )*self.target_size_wh_tensor

        zeros= tf.zeros_like(trans_xy[...,:1])
        ones= tf.ones_like(zeros)

        flatten_translation_matrix = tf.concat(
            [
                ones,  zeros, trans_xy[...,:1], 
                zeros,  ones,  trans_xy[...,1:2],
                zeros, zeros,  ones
            ],
            axis=-1
        ) #(4,9)
        translation_matrix = tf.reshape(
            flatten_translation_matrix, (-1,3,3)
        )
        scale = tf.random.uniform(
            shape=(batch_size,1),
            minval=self.scaling_ratio_range[0],
            maxval=self.scaling_ratio_range[1],
            dtype = self.compute_dtype

        )
        flatten_scaling_matrix = tf.concat(
            [scale, zeros, zeros, 
            zeros, scale, zeros,
            zeros,zeros,ones], axis=1
        )#(b,3,1)
        scaling_matrix = tf.reshape(
            flatten_scaling_matrix,[-1,3,3]
        )
        tf_matAffine_3x3 = tf.linalg.matmul(
                    translation_matrix,
                    scaling_matrix,
        )# (b,3,3)
        flatten_matrix_8x1 = tf.reshape(
            tf_matAffine_3x3, shape=(-1,9)
        )[:,:8]

        # in following ops : tfa.image.transform / tf.raw_ops.ImageProjectiveTransformV3,
        # data['image'] is tf.Tensor Type with shape (b,640,640,3) not Ragged Tensor
        if False :
            data['image'] = tfa.image.transform(
                data['image'],
                tf.cast(flatten_matrix_8x1, dtype=tf.float32),
                interpolation=self.interpolation.lower(),
                fill_mode='constant',
                output_shape = self.target_size 
            )#image_size_yx
        else:
            '''
            Note : tf.raw_ops.ImageProjectiveTransformV3 only support float32
            '''
            data["image"] = tf.raw_ops.ImageProjectiveTransformV3(
                images = data["image"],
                transforms = tf.cast( flatten_matrix_8x1, dtype=tf.float32),
                fill_value=tf.convert_to_tensor(0., tf.float32),
                fill_mode ="CONSTANT",
                interpolation = self.interpolation.upper(), 
                output_shape =  self.target_size                            
            )
        
        # tf_transform_inv = tf.linalg.inv(tf_matAffine_3x3) # (b,3,3)
        # matAffine_2x3 = tf_transform_inv[:,:2,:] #(b,2,3)
        tf_transform_inv = tf.linalg.inv(
            tf.cast(
                tf_matAffine_3x3, 
                dtype= self.compute_dtype if tf.__version__ > '2.10.0' else tf.float32
            )
        ) # (b,3,3)
        matAffine_2x3 = tf.cast( 
            tf_transform_inv[:,:2,:]  , dtype=self.compute_dtype
        )#(b,2,3)

        #----------------bbox_xywh------------
        bbox_xywh = data['bbox'] #(b,num_bbox,4)
        bbox_lt = bbox_xywh[...,:2]  #(b,num_bbox,2)
        bbox_rb = bbox_lt[...,:2] + bbox_xywh[...,2:] #(b,num_bbox,2)
        pts_xy = tf.stack([bbox_lt, bbox_rb], axis=-2)   #(b,num_bbox,2,2)

        new_pts_xy = self._ragged_xy_points_affine_transform(
            pts_xy,
            matAffine_2x3
        )# #(b,padded_num_bbox,2, 2)   @tensor
      
        bbox_xywh_tensor = tf.concat(
            [new_pts_xy[...,0,:], new_pts_xy[...,1,:] - new_pts_xy[...,0,:]], 
            axis=-1
        ) #(b,padded_num_bbox,4)  @tensor
        data['bbox'] = bbox_xywh_tensor
        if not self.use_kps :  
            return data  
        #-------------------------used_kps-----------------------------------
        kps = data['kps'] #(b,None,17,3)
        new_pts_xy = self._ragged_xy_points_affine_transform(
            kps[...,:2] ,
            matAffine_2x3
        )# (b,None,17, 2)
        kps = tf.concat(
            [new_pts_xy, kps[...,2:3]], axis=-1
        ) #(b,None,17,3)@tensor
            
        data['kps'] = kps
        return data
       
    def fast_rand_crop(
            self, data : Dict, batch_size : int
    ) ->Dict:
        #mosaic_lt_sampler
        mosaic_lt_xy = tf.random.uniform( shape=(batch_size,2),
            minval=self.lt_ratio_range[0], #0.25
            maxval=self.lt_ratio_range[1], #0.75
            dtype=self.compute_dtype,
        )*self.target_size_wh_tensor
       
        translation_xy = self.target_size_wh_tensor - mosaic_lt_xy #(b,2)

        cropping_boxes = tf.stack(
            [
                translation_xy[:,1],
                translation_xy[:,0],
                translation_xy[:,1]+self.target_size_wh_tensor[1], # self.target_size_wh_tensor[1]
                translation_xy[:,0]+self.target_size_wh_tensor[0], #self.target_size_wh_tensor[0]
            ],
            axis=-1
        )
        #cropping_boxes /= tf.tile(tf.reverse(self.target_size_wh_tensor*2-1,[0]),[2])
        cropping_boxes /= tf.tile(tf.reverse(self.target_size_wh_tensor*2-1,[0]), [2] )
        data["image"] = tf.image.crop_and_resize(
                data["image"],
                tf.cast(cropping_boxes, self.compute_dtype),
                tf.range(batch_size),
                self.target_size,     #self.target_size
                method=self.interpolation,
        )

        'transform from mosaic4x(1280,1280,3) coords to mosaic(640,640,3)'
        data['bbox'] = tf.concat(
            [data["bbox"][...,:2]-translation_xy[:,None,:],data["bbox"][...,2:] ],
            axis=-1
        ) 
   
        if self.use_kps :
            data['kps'] =  tf.concat(
                [data['kps'][...,:2] -translation_xy[:,None, None,:], data['kps'][...,2:3]],
                axis=-1
            )
        return data 
        

    @tf.function
    def call(self,dic : Dict,  *args, **kwargs) ->Dict:
        #data = {k:v for k,v in dic.items()}
        data = self._op_copy_dict_data(dic) #must copy data ???
        del dic
        if not isinstance(data,dict):
            raise TypeError(
                "data type must be dict[str,Tensor],"
                f" but got {type(data)}@{self.__class__.__name__}"
            )
        if self.use_kps :
            if data.get('kps',None) is None:
                raise ValueError(
                    "Mosaic settings is to use keypoints "
                    f"but cannot get data['kps'] @{self.__class__.__name__}"
                )
            
        tf.debugging.assert_equal(
            data['bbox_format'], "xywh", 
            message= f"bbox_format must be 'xywh', \
            but got {data['bbox_format']}@{self.__class__.__name__}"
        )

        if data['image'].shape.rank == 4:
            data = self.batched_transform(data)
        else:
            raise ValueError(
                "Mosaic must receive batched images, "
                f"but got {data['image'].shape.rank} @{self.__class__.__name__}"
                "Please call this transform by sending samples with batch_size=4 at least"
            )
        

        return data
 
    def _create_mosaic_data(
        self, data, batch_idx, use_kps=False
    ) -> Dict[str,tf.Tensor]:
        
        mosaic_data = dict()
        mosaic_data['image'] = tf.gather(
            data['image'], batch_idx
        )#(b,4,None,None,3)-RaggedTensor-uint8 => (b*4,None,None,3)
        mosaic_data['bbox'] = tf.gather(
            data['bbox'], batch_idx
        )#(b,4,Ragged_num_obj,4) -<RaggedTensor-float32>
        mosaic_data['labels']  = tf.gather(
            data['labels'], batch_idx
        ) #(b,4,None)-RaggedTensor
        if use_kps :
            mosaic_data['kps'] = tf.gather(
                data['kps'], batch_idx
            )#(b,4,None,4) -<RaggedTensor-float32>

        return mosaic_data
    
   
    def mosaic4x_data_split(self,tensor):
        return tensor[::4], tensor[1::4], tensor[2::4], tensor[3::4]   


    def batched_transform(self, data : Dict[str,Tensor])->Dict[str,Tensor]:
  
        #return self.non_mosaic_resize(data, self.use_kps)
        # 'no mosaic , only resize one image to target size'
        #if (self.prob < rand_prob()):
        # bool_op = self.rand_op()
        # data = tf.cond(bool_op, 
        #         lambda : self.non_mosaic_resize(data, self.use_kps),
        #         lambda : self.test(data)
        # )
        # if self.rand_op():    
        #     data = self.non_mosaic_resize(data, self.use_kps)
        #     return data
            #return data
        #else:
            

        img_dtype = data["image"].dtype
        batch_size=tf.shape( data["image"])[0]

        if  self.use_kps and type(data['kps'])!=type(data['bbox']):
            raise TypeError(
                f"type(data['kps']) must be same as type(data['bbox']) @{self.__class__.__name__}, "
                f"but got kps : {type(data['kps'])} and bbox :{type(data['bbox'])}"
            )

        'batch_idx for selecting images'
        batch_idx = tf.random.uniform( shape=(batch_size,3),
                minval=0,
                maxval=batch_size,
                dtype=tf.dtypes.int32,
        )
        batch_idx = tf.concat(
            [tf.expand_dims(tf.range(batch_size), axis=-1), batch_idx],
            axis=-1
        )
        'create mosaic data by selecting image from batch'
        mosaic_data = self._create_mosaic_data(
            data, batch_idx, self.use_kps
        )

        mosaic_data = self._combine_mosaic4(
            mosaic_data, batch_size
        )

        data["image"] = mosaic_data['image'].to_tensor()
        data["bbox"] = mosaic_data['bbox']
        data["labels"] = mosaic_data['labels']

        if data.get('kps',None)!=None :
            data["kps"] = mosaic_data['kps']



        if self.apply_affine_transform :
            data = self.affine_transform(data , batch_size) 
        else:
            data = self.fast_rand_crop(data , batch_size) 

        # # 'update image dtype to same dtype of input image'
        data["image"] = tf.cast(data["image"], dtype=img_dtype)

        # 'filter out bbox and kps oustsie generated mosiac images'
        data = self.filter_data_outside_mosaic_img(data)

        # 'filter out too small bbox in mosiac images'
        #return data
        # if self.min_bbox_area :
        #     data = self.small_bbox_data_filter(data)

        return self.small_bbox_data_filter(data) if self.min_bbox_area else data
    
    def filter_data_outside_mosaic_img(self,data):   
   
        bboxes_xy_lt = tf.math.maximum(
            x = data['bbox'][...,:2],
            y = 0.
        )
        bboxes_xy_rb = tf.math.minimum(
            x = data['bbox'][...,:2]+data['bbox'][...,2:],
            y = self.target_size_wh_tensor
        )
        bboxes_wh = bboxes_xy_rb - bboxes_xy_lt #(b,num_obj,2)
        #tf.print(tf.type_spec_from_value(data['bbox']))

        mask = tf.math.reduce_all(
            tf.greater(bboxes_wh,0.), axis=-1
        ) #(b,num_obj) , true mean valid, false is invalid @tf.bool
        data['bbox'] = tf.concat(
            [bboxes_xy_lt, bboxes_wh], axis=-1
        )
        
        if self.use_kps:
            
            pose_obj_mask = tf.ragged.boolean_mask(
                mask,
                tf.equal(data['labels'],self.pose_obj_label_id).with_row_splits_dtype('int64')
            )#(b, num_obj ) ->(b, num_pose_obj )   
            # pass
            if isinstance(data['kps'], tf.RaggedTensor):
                data['kps'] = tf.ragged.boolean_mask(
                    data['kps'], pose_obj_mask.with_row_splits_dtype('int64')
                ) #(b, num_pose_obj )         
            else:
                data['kps'] = tf.where(
                    pose_obj_mask[...,None,None], data['kps'], 0.
                )

            kpt_mask = tf.concat(
                [
                    tf.greater(data['kps'][...,:2],0.), 
                    tf.less(data['kps'][...,:2],self.target_size_wh_tensor),
                    tf.greater(data['kps'][...,2:3],0)
                ],
                axis= -1
            ) #(b,None,17, 2+2+1)=> (None,17,)
            kpt_mask = tf.math.reduce_all(
                kpt_mask, axis=-1
            ) #(n.None,17, 5)=> (n,None,17)
            data['kps'] = tf.where(
                kpt_mask[...,None], data['kps'], tf.zeros_like(data['kps'])
            )

        if isinstance(data['bbox'], tf.RaggedTensor):
            #tf.print(mask[0].shape, features['labels'][0].shape)
            data['bbox'] = tf.ragged.boolean_mask(
                data['bbox'], mask.with_row_splits_dtype('int64')
            )
            data['labels'] = tf.ragged.boolean_mask(
                data['labels'], mask.with_row_splits_dtype('int64')
            )
            
        else:
            data['bbox'] = tf.where(
                mask[...,None], data['bbox'], 0.
            )
            data['labels'] = tf.where(
                mask, data['labels'], self.label_padding_val
            )  

        # if data.get('gt_mask',None)is not None :
        #     data['gt_mask'] = mask
 
        return data

# class Mosaic:
#     VERSION = '1.1.0'
#     r""" Mosaic implements the mosaic data augmentation technique.
#     date : 2024/1/24
#     author : Dr. Shen 

#     to do : 1. add bbox format converter , to ensure all type input can work well
#             4. verify ImageProjectiveTransformV3
#             6. verify input is tensor not ragged tensor



#     Required Keys:
#         - image ----- TensorSpec(shape=(b, h, w, 3), dtype=tf.uint8, name=None)
#         - labels ----- RaggedTensorSpec(TensorShape([b, None]), tf.int32, 1, tf.int64) 
#         - bbox ----- RaggedTensorSpec(TensorShape([b, None, 4]), tf.float32, 1, tf.int64)
#         - kps(optional) ----- RaggedTensorSpec(TensorShape([b, None, num_joints, 3]), tf.float32, 1, tf.int64)

#     Modified Keys:
#         - image 
#         - labels
#         - bbox
#         - kps(optional)



#     Args:
#         target_size (tuple[int]): output dimension after mosaic transform, `[height, width]`.
#             If `None`, output is the same size as input image . defaults to (640,640).
#         interpolation (str): Interpolation mode. Supported values: "nearest","bilinear". Defaults to 'bilinear'
#         center_ratio_range (Tuple[float] ): 
#         scaling_ratio_range (Tuple[float]) : Min and max ratio of scaling  used in affine transform. 
#             if use_affine_transform = False, it is invalid, Defaults to (0.75, 1.25).
#         num_used_kps (int) : number of joints used for each pose object, Defaults to 0
#            for object detection training task,  num_used_kps = 0
#            for bottomup pose detection training task,  num_used_kps = 17 if using standard coco style keypoints
#         pose_obj_label_id (int) : the label index of pose object, Defaults to 1
#         img_padding_val (int) : padding value for each image
#         label_padding_val (int) : padding value for labels id corresponding to invalid bboxes 
#             invalid bboxes are outside mosaic image size. Defaults to 0
#         use_affine_transform (bool) : whether to use affine_transform to crop and resize image , Default to True
#         min_bbox_area (float) : minimum area of bbox for labeling, if min_bbox_area!=0., 
#             transform will filter out instances which of bboox area is greater than min_bbox_area. Defualt to 0
#         parallel_iterations (int) : The number of iterations allowed to run in parallel for tf.map_fn. Default to 32    
#         prob (float) : The probability to enable MixUp . Defaults to 0.5

        

#     References:
#         - [Yolov4  paper] (https://arxiv.org/pdf/2004.10934).
#         - [Yolov5 implementation](https://github.com/ultralytics/yolov5).
#         - [YoloX implementation](https://github.com/Megvii-BaseDetection/YOLOX)
#         - ['Mosaic' implement of Yolov8 ] (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py).
#         - ['Mosaic' implement of keras_cv] (https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing/mosaic.py).
#         - ['Mosaic' implement of mmdet] ( https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/transforms/transforms.py).

#     Sample usage:

#     ```python
#     import copy
#     from lib.Registers import DATASETS
#     from lib.datasets.transforms import Mosaic, MixUp
#     ---------------------------------------------------------------  
#     tfrec_val_dataset = dict(
#         type = 'Test_CocoStylePoseDataset_Parser', 
#         data_root = val_tfrec_bottmup_pose_dir,
#         is_whole_body_boxes = False,
#         sel_kpt_idx  = [i for i in range(17)],
#     )

#     mosaic = Mosaic( 
#         target_size = (640,640),
#         interpolation = 'bilinear',
#         center_ratio_range = (0.75, 1.25),
#         scaling_ratio_range= (0.75, 1.25),
#         num_used_kps = 17,
#         min_bbox_area = 0., #4096./2,
#         use_affine_transform= True,
#         prob = 1.
#     )

#     ---------------------------------------------------------------  
#     test_dataloader_cfg =  dict(
#         type = 'dataloader',
#         batch_size = 16,
#         prefetch_size = 4,
#         shuffle  =  True,
#         tfrec_datasets_list = [tfrec_val_dataset],
#         augmenters = [mosaic], 
#         codec = None
#     )

#     tfds_builder = DATASETS.build(
#         copy.deepcopy(test_dataloader_cfg)
#     )
#     batch_dataset = tfds_builder.GenerateTargets(
#         test_mode=False, 
#         unpack_x_y_sample_weight= False, 
#         ds_weights =None
#     )
#     --------------------------------------------------------------- 
#     tfds_builder.get_pipeline_cfg() 
#     batch_id = 0
#     for features in batch_dataset.take(1):
#         PackInputTensorTypeSpec(features,{}, show_log=True)
#         image = features["image"][batch_id].numpy()
#         labels = features['labels'][batch_id]
#         bboxes = features['bbox'][batch_id]
        
#         if features.get('kps',None) is not None :
#             kpts= features['kps'][batch_id] 
#             print(f" Shape = [labels : {labels.shape}, bboxes : {bboxes.shape}, kpts : {kpts.shape} ]")
#             pose_obj_num = tf.reduce_sum(
#                 tf.cast(labels==1, dtype=tf.int32)
#             )
#             print(f"labels : {labels}, pose_obj_num : {pose_obj_num}") 
        
#         plt.figure(figsize=(10, 10))
#         if  features.get('meta_info',None) is not None :
#             plt.title(
#                 f"image_id: {features['meta_info']['image_id'][batch_id]}"
#             )
#         plt.imshow(image)

#         ith_obj = -1
#         for i, (bbox,label) in enumerate(zip(bboxes,labels)):
#             if label==1:
#                 color = np.random.uniform(
#                     low=0., high=1., size=3
#                 )
#                 ith_obj +=1
#             ax = plt.gca()
#             x1, y1, w, h = bbox
#             patch = plt.Rectangle(
#                 [x1, y1], w, h, fill=False, edgecolor=color, linewidth=2
#             )
#             ax.add_patch(patch)
#             text = "{}".format(label)
#             ax.text(
#                 x1, y1, text,
#                 bbox={"facecolor": color, "alpha": 0.8},
#                 clip_box=ax.clipbox,
#                 clip_on=True,
#             )

#             if features.get('kps',None) is None :
#                 continue
#             kpt_ith = kpts[ith_obj]
#             for j in range(0,kpt_ith.shape[0]):
#                 kps_x = int((kpt_ith[j,0]))
#                 kps_y = int((kpt_ith[j,1]))
#                 plt.scatter(kps_x,kps_y, color=color)

#     """

#     def __init__(
#         self,
#         target_size : Union[Tuple,List] = (640,640),
#         interpolation : str = 'bilinear',
#         center_ratio_range : Tuple[float] = (0.75, 1.25),
#         scaling_ratio_range: Tuple[float, float] = (0.75, 1.25),
#         num_used_kps : int = 0,
#         pose_obj_label_id : int = 1,
#         img_padding_val : int = 0,
#         label_padding_val : int =  0,
#         use_affine_transform : bool = False,
#         min_bbox_area : Optional[float] = None,
#         parallel_iterations : int = 32,
#         prob: float = 0.
#     ):
#         'method'
#         self.apply_affine_transform = use_affine_transform
        
#         #bbox_padding_val
#         'basic cfg'

#         self.prob = prob
#         self.pose_obj_label_id = pose_obj_label_id
#         self.label_padding_val = label_padding_val
#         self.fill_value = img_padding_val
#         self.interpolation = interpolation
#         self.target_size = target_size
#         self.target_size_wh_tensor = tf.cast(
#             [target_size[1],target_size[0]], dtype=tf.float32
#         )
#         self.lt_ratio_range = (center_ratio_range[0]-0.5, center_ratio_range[1]-0.5)
#         self.scaling_ratio_range = scaling_ratio_range
#         self.min_bbox_area =  0. if min_bbox_area is None else min_bbox_area


#         'tf.map_fn'
#         self.parallel_iterations = parallel_iterations
#         self.num_used_kps = num_used_kps
#         self.use_kps = num_used_kps>0
#         #NonMosaic_output_signature
#         self.NonMosaic_output_signature = {
#             'image' : tf.TensorSpec((self.target_size[0],self.target_size[1],3),dtype=tf.float32) ,
#             'bboxes' : tf.RaggedTensorSpec( tf.TensorShape([None, 4]), ragged_rank=0, dtype=tf.float32)    
#         }
#         #Mosaic_output_signature
#         self.Mosaic_output_signature = {
#             'mosaic_images' :  tf.TensorSpec((self.target_size[0]*2,self.target_size[1]*2,3), dtype=tf.float32),
#             'mosaic_bboxes' : tf.RaggedTensorSpec(tf.TensorShape([None, 4]), ragged_rank=0, dtype=tf.float32),
#             'mosaic_labels' : tf.RaggedTensorSpec(tf.TensorShape([None]), ragged_rank=0, dtype=tf.int32),
#         }
#         if num_used_kps:
#             self.NonMosaic_output_signature.update(
#                 {'kps' : tf.RaggedTensorSpec(tf.TensorShape([None,num_used_kps,3]), ragged_rank=0, dtype=tf.float32)}
#             )
#             self.Mosaic_output_signature.update(
#                 {'mosaic_kps' : tf.RaggedTensorSpec(tf.TensorShape([None,num_used_kps,3]), ragged_rank=0, dtype=tf.float32)}
#             )

#     def padding_resize(
#         self,
#         image,
#         bboxes_xywh = None,
#         kps_xy_vis = None,
#         pad_type='lt'
#     ) ->dict:

#         src_img_shape = tf.shape(image)[:2]
#         image = tf.image.resize(
#             image.to_tensor(),
#             size=self.target_size,
#             method = self.interpolation,
#             preserve_aspect_ratio=True
#         )
#         resized_img_shape = tf.shape(image)[:2]
#         resize_ratio = tf.cast(
#             resized_img_shape[0]/src_img_shape[0], dtype=tf.float32
#         )
#         padding_y = self.target_size[0]-resized_img_shape[0] #int
#         padding_x = self.target_size[1]-resized_img_shape[1]

#         if pad_type=='lt' :
#             paddings = tf.cast( [[0,padding_y],[0,padding_x],[0,0]],dtype=tf.int32)
#             offset_xy = tf.cast( [0.,0.], dtype=tf.float32)
#         elif pad_type=='rt' :
#             paddings = tf.cast([[0,padding_y],[padding_x,0],[0,0]],dtype=tf.int32)
#             offset_xy = tf.cast( [padding_x, 0],dtype=tf.float32)
#         elif pad_type=='lb' :
#             paddings = tf.cast( [[padding_y,0],[0,padding_x],[0,0]],dtype=tf.int32)
#             offset_xy = tf.cast( [0, padding_y], dtype=tf.float32)
#         elif pad_type=='rb' :
#             paddings = tf.cast( [[padding_y,0],[padding_x,0],[0,0]],dtype=tf.int32)
#             offset_xy = tf.cast( [padding_x,padding_y], dtype=tf.float32)
#         elif pad_type=='center' :
#             'only use on non-mosaic'
#             half_padding_x = padding_x//2
#             half_padding_y = padding_y//2
#             paddings = tf.cast(
#                 [
#                     [half_padding_y, padding_y-half_padding_y],
#                     [half_padding_x, padding_x-half_padding_x],
#                     [0,0]
#                 ]
#                 ,dtype=tf.int32
#             )
#             offset_xy = tf.cast( 
#                 [half_padding_x,half_padding_y], dtype=tf.float32
#             )
#         else:
#             raise RuntimeError(
#                 f'unknow pad_type : {pad_type} , no support'
#             )

#         image = tf.pad(
#             image,
#             paddings, 
#             mode='CONSTANT', 
#             constant_values=self.fill_value
#         )

#         if bboxes_xywh is not None:
#             bboxes_xywh = bboxes_xywh*resize_ratio
#             bboxes_xywh = tf.concat(
#                 [bboxes_xywh[...,:2]+offset_xy, bboxes_xywh[...,2:]],
#                 axis=-1
#             )

#         if kps_xy_vis is not None:
#             kps_xy = kps_xy_vis[...,:2]*resize_ratio
#             kps_xy_vis = tf.concat(
#                 [kps_xy[...,:2]+offset_xy, kps_xy_vis[...,2:3]],
#                 axis=-1
#             )       
#         return{
#             "image" : image,
#             "bboxes" : bboxes_xywh,
#             "kps" :  kps_xy_vis
#         }


#     def combine_mosaic4(self, mosaic_data):
#         r''''
#         - input mosaic_data
#         mosaic_data['mosaic_images'] : (4,None,None,3) @RaggedTensor__uint8
#         mosaic_data['mosaic_bboxes'] : (4,None,4) @RaggedTensor__flaot32
#         mosaic_data['mosaic_labels'] : (4,None) @RaggedTensor__int32

#         - output mosaic_data
#         mosaic_data['mosaic_images'] : (640*2,640*2,3) @Tensorr__uint8
#         mosaic_data['mosaic_bboxes'] : (None,4) @RaggedTensor__flaot32
#         mosaic_data['mosaic_labels'] : (None,) @RaggedTensor__int32
#         '''
#         lt, rt, lb, rb = [
#             self.padding_resize(
#                 mosaic_data['mosaic_images'][i],
#                 mosaic_data['mosaic_bboxes'][i],
#                 mosaic_data['mosaic_kps'][i] if self.use_kps else None,
#                 pad_type=method
#             )for i, method in enumerate(['rb','lb','rt','lt'])
#         ] #padding_types = ['rb','lb','rt','lt']
#         #------------------------------mosaic_image-----------------------------------------
#         '''get mosaic4_image concat([(640,640,3), (640,640,3),(640,640,3),(640,640,3)] -> (1280,1280,3)'''
#         mosaic_data['mosaic_images'] = tf.concat(
#             [
#                 tf.concat([lt['image'], rt['image']], axis=1),  # upper_stack
#                 tf.concat([lb['image'], rb['image']], axis=1)   # lower_stack
#             ],
#             axis=0
#         )
#         #------------------mosaic_bboxes / mosaic_mask-----------------------------------------
#         mosaic_data['mosaic_bboxes'] = tf.concat(
#             [
#                 lt['bboxes'],
#                 rt['bboxes'] + tf.cast([self.target_size[1],0,0,0], dtype=tf.float32),
#                 lb['bboxes'] + tf.cast([0,self.target_size[0],0,0], dtype=tf.float32),
#                 rb['bboxes'] + tf.cast([self.target_size[1],self.target_size[0],0,0], dtype=tf.float32)
#             ],
#             axis=0
#         ) #(Ragged_num_obj,4) RaggedTensor

#         mosaic_mask= tf.less_equal(
#             x = tf.math.reduce_prod(mosaic_data['mosaic_bboxes'][...,2:], axis=-1),
#             y = 0.
#         )#(b,Ragged_num_obj) RaggedTensor

#         mosaic_data['mosaic_bboxes'] = tf.where(
#            mosaic_mask[...,None], 
#            0. , 
#            mosaic_data['mosaic_bboxes']
#         )
#         #------------------------------mosaic_labels-----------------------------------------
#         # mosaic_data['mosaic_labels'] = tf.where(
#         #     mosaic_mask, 
#         #     self.label_padding_val, 
#         #     mosaic_data['mosaic_labels'].merge_dims(0, 1)
#         # )
#         mosaic_data['mosaic_labels'] = mosaic_data['mosaic_labels'].merge_dims(0, 1)


#         if not self.use_kps :
#             return mosaic_data
        
#         #------------------------------mosaic_kps-----------------------------------------
#         mosaic_data['mosaic_kps'] = tf.concat(
#             [
#                 lt['kps'],
#                 rt['kps'] + tf.cast([self.target_size[1],0, 0], dtype=tf.float32),
#                 lb['kps'] + tf.cast([0,self.target_size[0], 0], dtype=tf.float32),
#                 rb['kps'] + tf.cast([self.target_size[1],self.target_size[0],0], dtype=tf.float32)
#             ],
#             axis=0
#         )#(None,17,3)
#         # get effective kps by vis!=0
#         mosaic_data['mosaic_kps'] = tf.where(
#             mosaic_data['mosaic_kps'][...,2:3]==0., 
#             0.,
#             mosaic_data['mosaic_kps']
#         )#(None,17,3)

#         return mosaic_data

    
#     def _ragged_xy_points_affine_transform(self, pt_xy, matAffine_2x3):
#         """ 
#         pt_xy           pt_xy                   associated
#         shape_rank      shape                   matAffine_2x3   
#         --------------------------------------- -------------------------------
#         3            (b, b1, b2, 2)             (b,None, None, 2,3)   
#         2            (b, b1, 2,)                (b,None, 2,3)       
#         1            (b,2)                      (b,2,3)  
#         """
#         # pt_shape_rank = pt_xy.shape.rank #(1,2,3)
#         # for _ in range (pt_shape_rank-2):
#         #     matAffine_2x3 = matAffine_2x3[:,None,...]
#         matAffine_2x3 = matAffine_2x3[:,None,None,...]   #(b,None, None, 2,3)   
 
#         pt_3x1 = tf.expand_dims( 
#             tf.concat(
#                 [pt_xy[...,:2], tf.ones_like(pt_xy[...,:1])],
#                 axis=-1
#             ) ,
#             axis=-1
#         ) #(points,3,1) 
#         #return pt_xy
    
#         new_pt = tf.map_fn(
#             fn=lambda x: tf.matmul(x[0],x[1]), 
#             elems=(matAffine_2x3,pt_3x1),
#             fn_output_signature = tf.RaggedTensorSpec(
#                 tf.TensorShape([None,None,2,1]), ragged_rank=0, dtype=tf.float32
#             ) 
#         ) #(b,num_obj,2,2,1) or (b,num_pose_obj,17,2,1)
#         #new_pt = tf.linalg.matmul(matAffine_2x3,pt_3x1, b_is_sparse=True) #(17,2,1)  (None,2,3)@(points,3,1)=  (None,3,1)
#         return tf.squeeze(new_pt, axis=-1)
    
#     def affine_transform(self, data,  batch_size=None) :

#         trans_xy = tf.random.uniform( 
#             shape=(batch_size,2),
#             minval=self.lt_ratio_range[0], #0.25
#             maxval=self.lt_ratio_range[1], #0.75
#             dtype=tf.dtypes.float32,
#         )*self.target_size_wh_tensor

#         zeros= tf.zeros_like(trans_xy[...,:1])
#         ones= tf.ones_like(zeros)

#         flatten_translation_matrix = tf.concat(
#             [
#                 ones,  zeros, trans_xy[...,:1], 
#                 zeros,  ones,  trans_xy[...,1:2],
#                 zeros, zeros,  ones
#             ],
#             axis=-1
#         ) #(4,9)
#         translation_matrix = tf.reshape(
#             flatten_translation_matrix, (-1,3,3)
#         )
#         scale = tf.random.uniform(
#                     shape=(batch_size,1),
#                     minval=self.scaling_ratio_range[0],
#                     maxval=self.scaling_ratio_range[1]
#         )
#         flatten_scaling_matrix = tf.concat(
#             [scale, zeros, zeros, 
#             zeros, scale, zeros,
#             zeros,zeros,ones], axis=1
#         )#(b,3,1)
#         scaling_matrix = tf.reshape(
#             flatten_scaling_matrix,[-1,3,3]
#         )
#         tf_matAffine_3x3 = tf.linalg.matmul(
#                     translation_matrix,
#                     scaling_matrix,
#         )# (b,3,3)
#         flatten_matrix_8x1 = tf.reshape(
#             tf_matAffine_3x3, shape=(-1,9)
#         )[:,:8]

#         # in following ops : tfa.image.transform / tf.raw_ops.ImageProjectiveTransformV3,
#         # data['image'] is tf.Tensor Type with shape (b,640,640,3) not Ragged Tensor
#         if True :
#             data['image'] = tfa.image.transform(data['image'],
#                 flatten_matrix_8x1,
#                 interpolation=self.interpolation.lower(),
#                 fill_mode='constant',
#                 output_shape = self.target_size 
#             )#image_size_yx
#         else:
#             data["image"] = tf.raw_ops.ImageProjectiveTransformV3(
#                 images = data["image"],
#                 transforms = flatten_matrix_8x1,
#                 fill_value=tf.convert_to_tensor(self.pad_val, tf.float32),
#                 fill_mode ="CONSTANT",
#                 interpolation = self.interpolation.upper(), 
#                 output_shape =  self.target_size                            
#             )
        
#         tf_transform_inv = tf.linalg.inv(tf_matAffine_3x3) # (b,3,3)
#         matAffine_2x3 = tf_transform_inv[:,:2,:] #(b,2,3)

#         #----------------bbox_xywh------------
#         bbox_xywh = data['bbox'] #(b,num_bbox,4)
#         bbox_lt = bbox_xywh[...,:2]  #(b,num_bbox,2)
#         bbox_rb = bbox_lt[...,:2] + bbox_xywh[...,2:] #(b,num_bbox,2)
#         pts_xy = tf.stack([bbox_lt, bbox_rb], axis=-2)   #(b,num_bbox,2,2)

#         new_pts_xy = self._ragged_xy_points_affine_transform(
#             pts_xy,
#             matAffine_2x3
#         )# #(b,padded_num_bbox,2, 2)   @tensor
      
#         bbox_xywh_tensor = tf.concat(
#             [new_pts_xy[...,0,:], new_pts_xy[...,1,:] - new_pts_xy[...,0,:]], 
#             axis=-1
#         ) #(b,padded_num_bbox,4)  @tensor
#         data['bbox'] = bbox_xywh_tensor
#         if not self.use_kps :  
#             return data  
#         #-------------------------used_kps-----------------------------------
#         kps = data['kps'] #(b,None,17,3)
#         new_pts_xy = self._ragged_xy_points_affine_transform(
#             kps[...,:2] ,
#             matAffine_2x3
#         )# (b,None,17, 2)
#         kps = tf.concat(
#             [new_pts_xy, kps[...,2:3]], axis=-1
#         ) #(b,None,17,3)@tensor
            
#         data['kps'] = kps
#         return data
       
#     def fast_rand_crop(
#             self, data : Dict, batch_size : int
#     ) ->Dict:
#         #mosaic_lt_sampler
#         mosaic_lt_xy = tf.random.uniform( shape=(batch_size,2),
#             minval=self.lt_ratio_range[0], #0.25
#             maxval=self.lt_ratio_range[1], #0.75
#             dtype=tf.dtypes.float32,
#         )*self.target_size_wh_tensor
       
#         translation_xy = self.target_size_wh_tensor - mosaic_lt_xy #(b,2)

#         cropping_boxes = tf.stack(
#             [
#                 translation_xy[:,1],
#                 translation_xy[:,0],
#                 translation_xy[:,1]+self.target_size_wh_tensor[1], # self.target_size_wh_tensor[1]
#                 translation_xy[:,0]+self.target_size_wh_tensor[0], #self.target_size_wh_tensor[0]
#             ],
#             axis=-1
#         )
#         #cropping_boxes /= tf.tile(tf.reverse(self.target_size_wh_tensor*2-1,[0]),[2])
#         cropping_boxes /= tf.tile(tf.reverse(self.target_size_wh_tensor*2-1,[0]), [2] )
#         data["image"] = tf.image.crop_and_resize(
#                 data["image"],
#                 tf.cast(cropping_boxes, tf.float32),
#                 tf.range(batch_size),
#                 self.target_size,     #self.target_size
#                 method=self.interpolation,
#         )

#         'transform from mosaic4x(1280,1280,3) coords to mosaic(640,640,3)'
#         data['bbox'] = tf.concat(
#             [data["bbox"][...,:2]-translation_xy[:,None,:],data["bbox"][...,2:] ],
#             axis=-1
#         ) 
   
#         if self.use_kps :
#             data['kps'] =  tf.concat(
#                 [data['kps'][...,:2] -translation_xy[:,None, None,:], data['kps'][...,2:3]],
#                 axis=-1
#             )
#         return data 
        

#     @tf.function
#     def __call__(self,dic : Dict) ->Dict:
#         data = {k:v for k,v in dic.items()}
#         del dic
#         if not isinstance(data,dict):
#             raise TypeError(
#                 "data type must be dict[str,Tensor],"
#                 f" but got {type(data)}@{self.__class__.__name__}"
#             )
#         if self.use_kps :
#             if data.get('kps',None) is None:
#                 raise ValueError(
#                     "Mosaic settings is to use keypoints "
#                     f"but cannot get data['kps'] @{self.__class__.__name__}"
#                 )
            
#         tf.debugging.assert_equal(
#             data['bbox_format'], "xywh", 
#             message= f"bbox_format must be 'xywh', \
#             but got {data['bbox_format']}@{self.__class__.__name__}"
#         )

#         if data['image'].shape.rank == 4:
#             data = self.transform(data)
#         else:
#             raise ValueError(
#                 "Mosaic must receive batched images, "
#                 f"but got {data['image'].shape.rank} @{self.__class__.__name__}"
#                 "Please call this transform by sending samples with batch_size=4 at least"
#             )
#         return data
 
#     def _create_mosaic_data(
#         self, data, batch_idx, use_kps=False
#     ) -> Dict[str,tf.Tensor]:
        
#         mosaic_data = dict()
#         mosaic_data['mosaic_images'] = tf.gather(
#             data['image'], batch_idx
#         )#(b,4,None,None,3)-RaggedTensor-uint8
#         mosaic_data['mosaic_bboxes'] = tf.gather(
#             data['bbox'], batch_idx
#         )#(b,4,Ragged_num_obj,4) -<RaggedTensor-float32>
#         mosaic_data['mosaic_labels']  = tf.gather(
#             data['labels'], batch_idx
#         ) #(b,4,None)-RaggedTensor
#         if use_kps :
#             mosaic_data['mosaic_kps'] = tf.gather(
#                 data['kps'], batch_idx
#             )#(b,4,None,4) -<RaggedTensor-float32>
#         return mosaic_data
    
#     def non_mosaic_resize(self, data, use_kps):

#         non_mosaic_data = tf.map_fn(
#             lambda x : self.padding_resize(x[0], x[1], x[2], 'center'),
#             (data['image'] , data['bbox'], data['kps'] if use_kps else None),
#             parallel_iterations = self.parallel_iterations,
#             fn_output_signature = self.NonMosaic_output_signature
#         )
#         data['image'] = tf.cast(
#             non_mosaic_data['image'], dtype=data['image'].dtype
#         )
#         data['bbox'] = non_mosaic_data['bboxes']

#         if  use_kps :
#             data["kps"] = non_mosaic_data['kps']

#         return data
    
#     def transform(self, data : Dict[str,Tensor])->Dict[str,Tensor]:

#         'no mosaic , only resize one image to target size'
#         if self.prob < rand_prob():
#             return self.non_mosaic_resize(data, self.use_kps)

#         img_dtype = data["image"].dtype
#         batch_size=tf.shape( data["image"])[0]

#         if  self.use_kps and type(data['kps'])!=type(data['bbox']):
#             raise TypeError(
#                 f"type(data['kps']) must be same as type(data['bbox']) @{self.__class__.__name__}, "
#                 f"but got kps : {type(data['kps'])} and bbox :{type(data['bbox'])}"
#             )

#         'batch_idx for selecting images'
#         batch_idx = tf.random.uniform( shape=(batch_size,3),
#                 minval=0,
#                 maxval=batch_size,
#                 dtype=tf.dtypes.int32,
#         )
#         batch_idx = tf.concat(
#             [tf.expand_dims(tf.range(batch_size), axis=-1), batch_idx],
#             axis=-1
#         )
#         'create mosaic data by selecting image from batch'
#         mosaic_data = self._create_mosaic_data(
#             data, batch_idx, self.use_kps
#         )
#         'combime mosaic data'
#         mosaic_data = tf.map_fn(
#             self.combine_mosaic4,
#             mosaic_data,
#             parallel_iterations = self.parallel_iterations,
#             fn_output_signature =  self.Mosaic_output_signature
#         )
        
#         data["image"] = mosaic_data['mosaic_images']
#         data["bbox"] = mosaic_data['mosaic_bboxes']
#         data["labels"] = mosaic_data['mosaic_labels']
#         if self.use_kps :
#             data["kps"] = mosaic_data['mosaic_kps']
#         del mosaic_data

#         if self.apply_affine_transform :
#             data = self.affine_transform(data , batch_size) 
#         else:
#             data = self.fast_rand_crop(data , batch_size) 

#         # 'update image dtype to same dtype of input image'
#         data["image"] = tf.cast(data["image"], dtype=img_dtype)

#         'filter out bbox and kps oustsie generated mosiac images'
#         data = self.filter_data_outside_mosaic_img(data)

#         'filter out too small bbox in mosiac images'
#         # if self.min_bbox_area :
#         #     data = self.small_bbox_data_filter(data)
#         return self.small_bbox_data_filter(data) if self.min_bbox_area else data
    
#     def filter_data_outside_mosaic_img(self,data):   
   
#         bboxes_xy_lt = tf.math.maximum(
#             x = data['bbox'][...,:2],
#             y = 0.
#         )
#         bboxes_xy_rb = tf.math.minimum(
#             x = data['bbox'][...,:2]+data['bbox'][...,2:],
#             y = self.target_size_wh_tensor
#         )
#         bboxes_wh = bboxes_xy_rb - bboxes_xy_lt #(b,num_obj,2)
#         #tf.print(tf.type_spec_from_value(data['bbox']))

#         mask = tf.math.reduce_all(
#             tf.greater(bboxes_wh,0.), axis=-1
#         ) #(b,num_obj) , true mean valid, false is invalid @tf.bool
#         data['bbox'] = tf.concat(
#             [bboxes_xy_lt, bboxes_wh], axis=-1
#         )
        
#         if self.use_kps:
#             pose_obj_mask = tf.ragged.boolean_mask(
#                 mask,
#                 tf.equal(data['labels'],self.pose_obj_label_id).with_row_splits_dtype('int64')
#             )#(b, num_obj ) ->(b, num_pose_obj )   
#             # pass
#             if isinstance(data['kps'], tf.RaggedTensor):
#                 data['kps'] = tf.ragged.boolean_mask(
#                     data['kps'], pose_obj_mask.with_row_splits_dtype('int64')
#                 ) #(b, num_pose_obj )         
#             else:
#                 data['kps'] = tf.where(
#                     pose_obj_mask[...,None,None], data['kps'], 0.
#                 )

#             kpt_mask = tf.concat(
#                 [
#                     tf.greater(data['kps'][...,:2],0.), 
#                     tf.less(data['kps'][...,:2],self.target_size_wh_tensor),
#                     tf.greater(data['kps'][...,2:3],0)
#                 ],
#                 axis= -1
#             ) #(b,None,17, 2+2+1)=> (None,17,)
#             kpt_mask = tf.math.reduce_all(
#                 kpt_mask, axis=-1
#             ) #(n.None,17, 5)=> (n,None,17)
#             data['kps'] = tf.where(
#                 kpt_mask[...,None], data['kps'], tf.zeros_like(data['kps'])
#             )

#         if isinstance(data['bbox'], tf.RaggedTensor):
#             #tf.print(mask[0].shape, features['labels'][0].shape)
#             data['bbox'] = tf.ragged.boolean_mask(
#                 data['bbox'], mask.with_row_splits_dtype('int64')
#             )
#             data['labels'] = tf.ragged.boolean_mask(
#                 data['labels'], mask.with_row_splits_dtype('int64')
#             )
#         else:
#             data['bbox'] = tf.where(
#                 mask[...,None], data['bbox'], 0.
#             )
#             data['labels'] = tf.where(
#                 mask, data['labels'], self.label_padding_val
#             )  

#         if data.get('gt_mask',None)is not None :
#             data['gt_mask'] = mask
 
#         return data

#     def small_bbox_data_filter(self, data):
#         area = tf.reduce_prod(data['bbox'][...,2:],axis=-1) 
#         obj_mask = tf.greater(area, self.min_bbox_area) #(b,num_obj)

#         if self.use_kps :
#             pose_obj_mask = tf.ragged.boolean_mask(
#                 obj_mask,
#                 tf.equal(data['labels'], self.pose_obj_label_id).with_row_splits_dtype('int64')
#             )#(b, num_obj ) ->(b, num_pose_obj ) 
#             'pose_det'
#             data['kps'] = tf.ragged.boolean_mask(
#                 data['kps'], pose_obj_mask.with_row_splits_dtype('int64')
#             ) #(b, num_pose_obj )  

#         'obj_det'
#         data['bbox'] = tf.ragged.boolean_mask(
#             data['bbox'], 
#             obj_mask.with_row_splits_dtype('int64')
#         )
#         data['labels'] = tf.ragged.boolean_mask(
#             data['labels'], obj_mask.with_row_splits_dtype('int64')
#         )
#         return data

