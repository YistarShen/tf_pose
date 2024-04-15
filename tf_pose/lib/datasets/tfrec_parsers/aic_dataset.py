from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union
import tensorflow as tf
from lib.utils.common import is_path_available
from lib.Registers import DATASETS
from .base_parser import BaseParser
from .parser_utils import review_data_spec

@DATASETS.register_module()
class Parser_AicSinglePoseTFRec(BaseParser):
    VERSION = '1.0.0'
    r"""BaseCocoStyleDataset_Parser
    to parse tfrec dataset of coco kps, and only extract basic 17kps
    feature_description is same as coco official annotation .json file

    Args:
        data_root (str): tfrrec_data_dir, Default: ''.

    
    DATA_FORMAT_DICT={
        'image': [None,None,3],      # Scalar elements, no padding.
        'bbox': [4],           # Vector elements, padded to longest.
        'kps': [None,3],           # Vector elements, padded to longest.
        'image_size' : [2],
        'meta_info' : { 'src_image' :[None,None,3],
                        'src_height' : [],
                        'src_width' : [],
                        'src_keypoints':[None],
                        'src_bbox' : [4],
                        'src_num_keypoints': [], 
                        'area': [], 
                        'category_id' : [],
                        'id' : [],
                        'image_id' : [],
        },
        'transform2src' : { 'scale_xy' : [2],
                            'pad_offset_xy' : [2],
                            'bbox_lt_xy' : [2],
        }
    }

    """


    SUPPORT_POSE_STYLE = ['coco']
    def __init__(
            self,
        data_root : Optional[str]=None,
        pose_style : str = 'coco'
    ):
        super().__init__(data_root = data_root)
        
        self.feature_description = {
      
            "image": tf.io.FixedLenFeature([], tf.string),
            "bbox": tf.io.VarLenFeature(tf.float32), #xywh
            "category_id": tf.io.FixedLenFeature([], tf.int64),
            "id": tf.io.FixedLenFeature([], tf.int64),
            "image_id": tf.io.FixedLenFeature([], tf.int64),
            "num_keypoints": tf.io.FixedLenFeature([], tf.int64),
            "keypoints": tf.io.VarLenFeature(tf.float32),
            "img_url": tf.io.FixedLenFeature([], tf.string),
            "img_height": tf.io.FixedLenFeature([], tf.int64),
            "img_width": tf.io.FixedLenFeature([], tf.int64)
        }
        if pose_style.lower() not in self.SUPPORT_POSE_STYLE:
            ValueError(
                f"only pose_style='coco', but got {pose_style.lower()} @{self.__class__.__name__}, "
            ) 

        self.pose_style = pose_style.lower() 
        if self.pose_style  == 'coco':
            self.map2kpts_id = tf.constant( [3,0,4,1,5,2,9,6,10,7,11,8],dtype=tf.int32) 

    def review_data_spec(
            self, reviews : int = 1, plot_img : bool=True
    ):
        
        import matplotlib.pyplot as plt
        dataset  = self.gen_tfds(
            batch_size = 0, meta_info=True
        )
        review_data_spec(dataset, reviews, plot_img)
        del dataset

    def get_TotalSamples(self, 
                        log :bool = False) ->int:
        """ get TotalSamples of all tfrec files 
        it's very important to determine how many steps per epcoh for training

        Args:
            log (bool): print samples each file and total samples
        Returns:
            total_samples (int): total samples of all tfrec files
        """
        import re
        tfrec_files = tf.io.gfile.glob(f"{self.tfrecords_dir}/*.tfrec")
        total_samples =0
        for name in tfrec_files:
            name = name.replace(self.tfrecords_dir+"/",'')
            string = re.sub("[^0-9]", "", name)
            samples_in_tfrec = int(string[2:])
            if log :print(f"samples_in_tfrec : {samples_in_tfrec} @ <{name}>")

            total_samples += samples_in_tfrec
        del tfrec_files
        if log : print("num_TotalSamples:", total_samples)
        return total_samples

    
    def _kpts_map_to_coco_style(self,keypoints):
        # if tf.rank(keypoints)!=(2 or 3) :
        #     ValueError(
        #         f"keypoints.rank must be 2 or 3, but got {tf.shape(keypoints).rank} @{self.__class__.__name__}, "
        #     )  
        
        keypoints = tf.gather(
            keypoints, self.map2kpts_id, axis= -2
        ) #(None,12,3)
        keypoints = tf.concat(
            [tf.zeros_like(keypoints[...,:5,:]), keypoints], axis=-2
        )
        return keypoints
    
    def _kpts_map_to_aic_style(self,keypoints):
        return keypoints
    
    def _kpts_map_to_customized_style(self,keypoints):
        raise NotImplementedError()
    
    def __call__(self, 
                example : dict,
                meta_info: bool=False):
        
        example = tf.io.parse_single_example(
            example, self.feature_description
        )
        example["image"] = tf.io.decode_jpeg(
            example["image"], channels=3
        )
        example["bbox"] = tf.sparse.to_dense(example["bbox"])   
        example["bbox"] = tf.reshape(
            example["bbox"], shape=(4,)
        ) 
        img_shape_yx = tf.stack(
            [example["img_height"],example["img_width"]]
        )
        example["keypoints"] = tf.sparse.to_dense(
            example["keypoints"]
        )  
        'mapping to coco style'
        keypoints = tf.reshape(
            example["keypoints"], shape=(14,3)
        )

        keypoints = self._kpts_map_to_coco_style(keypoints)
        num_keypoints = tf.math.count_nonzero(keypoints[:,2])

        data = {
            'image' : example["image"], 
            'bbox' : example["bbox"], 
            'kps' : keypoints, 
            'image_size': tf.cast(img_shape_yx,dtype=tf.int32),
            'bbox_format' : tf.cast("xywh", dtype=tf.string)
        }
        
        if meta_info:
            ann = { 
                'src_image' : example["image"],
                'src_height' : example["img_height"],
                'src_width' : example["img_width"],
                'src_keypoints': keypoints,
                'src_bbox' : example["bbox"],
                'src_num_keypoints': num_keypoints, 
                'area': tf.math.reduce_prod(example["bbox"][2:]), 
                'category_id' : example["category_id"],
                'id' : example["id"],
                'image_id' : example["image_id"],
                'iscrowd' : tf.cast(0, dtype=tf.int64)
            }  
            transform2src = { 
                'scale_xy' : tf.constant([1.,1.],dtype=tf.float32),
                'pad_offset_xy' : tf.constant([0.,0.],dtype=tf.float32),
                'bbox_lt_xy' : example['bbox'][:2],
            }
            data['meta_info'] = ann
            data['transform2src'] = transform2src

        return data


@DATASETS.register_module()
class Parser_AicMultiPoseTFRec(Parser_AicSinglePoseTFRec):

    VERSION = '1.0.0'
    r"""CocoStylePoseDataset_Parser
        to parse tfrecord dataset of coco personal bboxes, face and two hands boxes

    Args:
        data_root (str): tfrec_data_dir

    Data Key Description :
        "image":   model input image shape=(None, None, 3)
        "bboxes":  boxes with xywg format, shape=(None, 4)
        "labels":  labels for bboxes, shape=(None,)
        "image_id": 
        "img_url": 
        "img_height": 
        "img_width": 
    
        
    parser_aic.explore_samples(
        take_batch_num = 2, 
        batch_size = 10,
        use_ragged_batch = True, 
        meta_info = True,
        batch_ids = [0],
        plot_transformed_bbox = True,
        figsize = (20,10)
    )
    """
    SUPPORT_POSE_STYLE = ['coco','aic']
    def __init__(
        self, 
        data_root : Optional[str]=None,
        sel_kpt_idx : List[int] = [i for i in range(17)],
        *args, **kwargs
    ):
        super().__init__(data_root = data_root,  *args, **kwargs)

        self.feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "id": tf.io.VarLenFeature(tf.int64),
            "iscrowd": tf.io.VarLenFeature(tf.int64),
            # bboxes
            "area": tf.io.VarLenFeature(tf.float32),
            "bboxes": tf.io.VarLenFeature(tf.float32),
            "labels": tf.io.VarLenFeature(tf.int64),
            # keypoints
            "keypoints": tf.io.VarLenFeature(tf.float32),
            "num_keypoints": tf.io.VarLenFeature(tf.int64),
            # image info
            "image_id": tf.io.FixedLenFeature([], tf.int64),
            "img_url": tf.io.FixedLenFeature([], tf.string),
            "img_height": tf.io.FixedLenFeature([], tf.int64),
            "img_width": tf.io.FixedLenFeature([], tf.int64),
        }

    def _kpts_map_to_coco_style(self,keypoints):
        #if tf.rank(keypoints)!=(2 or 3) :
        # if keypoints.shape.rank != (2 or 3) :
        #     ValueError(
        #         f"keypoints.rank must be 2 or 3, but got {tf.shape(keypoints).rank} @{self.__class__.__name__}, "
        #     )  
        
        keypoints = tf.gather(
            keypoints, self.map2kpts_id, axis= -2
        ) #(None,12,3)
        keypoints = tf.concat(
            [tf.zeros_like(keypoints[...,:5,:]), keypoints], axis=-2
        )
        return keypoints
    
    def __call__(
        self,  features : dict, meta_info: bool=False
    ):
        features = tf.io.parse_single_example(
            features, self.feature_description
        ) 
        features["image"] = tf.io.decode_jpeg(
            features["image"], channels=3
        )#(h,w,3)   
        img_shape_yx = tf.stack(
            [features["img_height"],features["img_width"]],
            name = 'img_shape_hw'
        )
        # basic bboxes and labels    
        features["bboxes"] = tf.sparse.to_dense(
            features["bboxes"]
        )
        features["bboxes"] = tf.reshape(
            features["bboxes"], shape=[-1,4] 
        )#(num_obj,4)
        features["labels"] = tf.cast( 
            tf.sparse.to_dense(features["labels"]), dtype=tf.int32
        )#(num_obj,)
        keypoints = tf.sparse.to_dense(
            features["keypoints"]
        )
        keypoints = tf.reshape(
            keypoints, shape=(-1,14,3)
        )

        if self.pose_style == 'coco':
            keypoints = self._kpts_map_to_coco_style(keypoints)
        else:
            keypoints = self._kpts_map_to_aic_style(keypoints)

        data = {
            'image' : features["image"],
            'kps' : keypoints,
            'bbox' : features["bboxes"],
            'labels' : features["labels"],
            'image_size': tf.cast(img_shape_yx,dtype=tf.int32),
            'bbox_format' : tf.cast("xywh", dtype=tf.string),
        }

        if meta_info:
            ann = {
                'src_image' : features["image"],
                'src_height' : features["img_height"],
                'src_width' : features["img_width"],
                'src_bbox' : features["bboxes"],
                'src_labels' :features['labels'] ,
                'src_keypoints' : keypoints,
                'src_num_keypoints' : tf.cast( 
                    tf.sparse.to_dense(features['num_keypoints']) , dtype=tf.int32
                ),
                'area':  tf.math.reduce_prod(features["bboxes"][...,2:]),
                'id' : tf.sparse.to_dense(features["id"]) ,
                'image_id' : features["image_id"],
                'iscrowd' : tf.cast( 
                    tf.sparse.to_dense(features["iscrowd"]), dtype=tf.int32
                )#(num_obj,)features["iscrowd"] 
            }
            data['meta_info'] = ann
        return data
    
