from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union
import tensorflow as tf
from lib.Registers import DATASETS
from .base_parser import BaseParser
from .parser_utils import review_data_spec



# -------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------
r""" 
BaseCocoStyle structure: type=dict str={
    'image': TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name=None), 
    'bbox': TensorSpec(shape=(4,), dtype=tf.float32, name=None), 
    'kps': TensorSpec(shape=(17, 3), dtype=tf.float32, name=None), 
    'image_size': TensorSpec(shape=(2,), dtype=tf.int32, name=None), 
    'bbox_format': TensorSpec(shape=(), dtype=tf.string, name=None), 
    'meta_info': {'src_image': TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name=None), 
                  'src_height': TensorSpec(shape=(), dtype=tf.int64, name=None), 
                  'src_width': TensorSpec(shape=(), dtype=tf.int64, name=None), 
                  'src_keypoints': TensorSpec(shape=(None,), dtype=tf.float32, name=None), 
                  'src_bbox': TensorSpec(shape=(4,), dtype=tf.float32, name=None), 
                  'src_num_keypoints': TensorSpec(shape=(), dtype=tf.int64, name=None), 
                  'area': TensorSpec(shape=(), dtype=tf.float32, name=None), 
                  'category_id': TensorSpec(shape=(), dtype=tf.int64, name=None), 
                  'id': TensorSpec(shape=(), dtype=tf.int64, name=None), 
                  'image_id': TensorSpec(shape=(), dtype=tf.int64, name=None), 
                  'iscrowd': TensorSpec(shape=(), dtype=tf.int64, name=None)
    }, 
    'transform2src': {'scale_xy': TensorSpec(shape=(2,), dtype=tf.float32, name=None), 
                  'pad_offset_xy': TensorSpec(shape=(2,), dtype=tf.float32, name=None), 
                  'bbox_lt_xy': TensorSpec(shape=(2,), dtype=tf.float32, name=None)
    }
}
"""

@DATASETS.register_module()
class Parser_CocoSinglePoseTFRec(BaseParser):
    VERSION = '1.0.1'
    r"""Parser_CocoTopdownTfrec_CocoPoseStyle
    to parse tfrecord dataset of coco kps, and only extract basic 17kps
    feature_description is same as coco offical annotation .json file

    Args:
        data_root (str): trrec_data_dir, Default: ''.

    

    """

    def __init__( 
            self, 
            data_root : Optional[str]=None,
            pose_style : str = 'coco'
    ):
        super().__init__(data_root = data_root )

        self.feature_description = {
      
            "image": tf.io.FixedLenFeature([], tf.string),
            "area": tf.io.FixedLenFeature([], tf.float32),
            "bbox": tf.io.VarLenFeature(tf.float32),
            "category_id": tf.io.FixedLenFeature([], tf.int64),
            "id": tf.io.FixedLenFeature([], tf.int64),
            "image_id": tf.io.FixedLenFeature([], tf.int64),

            "iscrowd": tf.io.FixedLenFeature([], tf.int64),
            "num_keypoints": tf.io.FixedLenFeature([], tf.int64),
            "keypoints": tf.io.VarLenFeature(tf.float32),

            "foot_kpts": tf.io.VarLenFeature(tf.float32),
            "foot_valid": tf.io.FixedLenFeature([], tf.int64),

            "face_box": tf.io.VarLenFeature(tf.float32),
            "face_kpts": tf.io.VarLenFeature(tf.float32),
            "face_valid": tf.io.FixedLenFeature([], tf.int64),

            "lefthand_box": tf.io.VarLenFeature(tf.float32),
            "lefthand_kpts": tf.io.VarLenFeature(tf.float32),
            "lefthand_valid": tf.io.FixedLenFeature([], tf.int64),

            "righthand_box": tf.io.VarLenFeature(tf.float32),
            "righthand_kpts": tf.io.VarLenFeature(tf.float32),
            "righthand_valid": tf.io.FixedLenFeature([], tf.int64),

            "img_url": tf.io.FixedLenFeature([], tf.string),
            "img_height": tf.io.FixedLenFeature([], tf.int64),
            "img_width": tf.io.FixedLenFeature([], tf.int64),
        }

        self.pose_style = pose_style.lower() 
        if self.pose_style not in ['coco']:
            raise ValueError(
                f"current version only support output kps with coco style define {self.__class__.__name__}"
            )

        
        
    def get_TotalSamples(self, log :bool = False) ->int:
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
    
    def review_data_spec(
            self, reviews : int = 1, plot_img : bool=True
    ):
        
        import matplotlib.pyplot as plt
        dataset  = self.gen_tfds(
            batch_size = 0, meta_info=True
        )
        review_data_spec(dataset, reviews, plot_img)
        del dataset

    def _kpts_map_to_coco_style(self,keypoints):
        return keypoints
    
    def _kpts_map_to_customized_style(self,keypoints):
        raise NotImplementedError()

    def __call__(
            self, example : dict, meta_info: bool=False
    ):
        
        example = tf.io.parse_single_example(
            example, self.feature_description
        )
        example["image"] = tf.io.decode_jpeg(
            example["image"], channels=3
        )
        example["bbox"] = tf.sparse.to_dense(example["bbox"])   
        example["bbox"] = tf.reshape(example["bbox"], shape=(4,)) 
        
        example["keypoints"] = tf.sparse.to_dense(example["keypoints"])  
        keypoints  = tf.reshape(example["keypoints"], shape=(17,3))

        if self.pose_style == 'coco':
            keypoints = self._kpts_map_to_coco_style(keypoints)
        else:
            keypoints = self._kpts_map_to_coco_style(keypoints)



        img_shape_yx = tf.stack([example["img_height"],example["img_width"]])
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
                'src_num_keypoints': example['num_keypoints'], 
                'area': example['area'], 
                'category_id' : example["category_id"],
                'id' : example["id"],
                'image_id' : example["image_id"],
                'iscrowd' : example["iscrowd"],
            }  

            transform2src = { 
                'scale_xy' : tf.constant([1.,1.],dtype=tf.float32),
                'pad_offset_xy' : tf.constant([0.,0.],dtype=tf.float32),
                'bbox_lt_xy' : example['bbox'][:2],
            }

            data['meta_info'] = ann
            data['transform2src'] = transform2src

        return data


# -------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------
@DATASETS.register_module()
class Parser_CocoMultiPoseTFRec(Parser_CocoSinglePoseTFRec):

    VERSION = '1.0.0'
    r"""CocoStylePoseDataset_Parser
        to parse tfrecord dataset of coco personal bboxes, face and two hands boxes

    Args:
        data_root (str): trrec_data_dir

    Data Key Description :
        "image":   model input image hape=(None, None, 3)
        "bboxes":  boxes with xywg format, shape=(None, 4)
        "labels":  labels for bboxes, shape=(None,)
        "image_id": 
        "img_url": 
        "img_height": 
        "img_width": 

    """
    def __init__(
        self, 
        data_root : Optional[str]=None,
        is_whole_body_boxes : bool = False,
        sel_kpt_idx : List[int] = [i for i in range(17)],
        *args, **kwargs
    ):
        super().__init__(data_root = data_root,  *args, **kwargs)
        self.is_whole_body_boxes = is_whole_body_boxes

        'whether to parse keypoints'
        if sel_kpt_idx is None :
            sel_kpt_idx = []
        self.sel_kpt_idx = sel_kpt_idx
        self.use_kps = True if len(self.sel_kpt_idx)>0 else False
        
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
            # foot/face/hand valid
            "foot_valid": tf.io.VarLenFeature(tf.int64),
            "face_valid": tf.io.VarLenFeature(tf.int64),
            "lefthand_valid": tf.io.VarLenFeature(tf.int64),
            "righthand_valid": tf.io.VarLenFeature(tf.int64),
            # image info
            "image_id": tf.io.FixedLenFeature([], tf.int64),
            "img_url": tf.io.FixedLenFeature([], tf.string),
            "img_height": tf.io.FixedLenFeature([], tf.int64),
            "img_width": tf.io.FixedLenFeature([], tf.int64),
        }

    def __call__(
        self, features : dict, meta_info: bool=False
    ):
        ''
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
        features["iscrowd"] = tf.sparse.to_dense(
            features["iscrowd"]
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

        if not self.is_whole_body_boxes :
            features['bboxes'] = tf.gather(
                features['bboxes'],
                tf.squeeze(tf.where(features['labels']==1), axis=-1),
                axis=0
            )
            features['labels'] = tf.gather(
                features['labels'],
                tf.squeeze(tf.where(features['labels']==1), axis=-1),
                axis=0
            )

        data = {
            'image' : features["image"],
            'bbox' : features["bboxes"],
            'labels' : features["labels"],
            'image_size': tf.cast(img_shape_yx,dtype=tf.int32),
            'bbox_format' : tf.cast("xywh", dtype=tf.string),
        }

        if self.use_kps :
            features["keypoints"] = tf.sparse.to_dense(
                features["keypoints"]
            )
            features["keypoints"] = tf.reshape(
                features["keypoints"], shape=(-1,133,3)
            )
            features["keypoints"] = tf.gather(
                features["keypoints"], self.sel_kpt_idx, axis=1
            )
            if self.pose_style == 'coco':
                features["keypoints"]  = self._kpts_map_to_coco_style(features["keypoints"] )
            else:
                features["keypoints"]  = self._kpts_map_to_coco_style(features["keypoints"] )
                
            data['kps'] = features["keypoints"]


        if meta_info:
            ann = {
                'src_image' : features["image"],
                'src_height' : features["img_height"],
                'src_width' : features["img_width"],
                'src_bbox' : features["bboxes"],
                'area':  tf.sparse.to_dense(features['area']),
                'src_labels' :features['labels'] ,
                'id' : tf.sparse.to_dense(features["id"]) ,
                'image_id' : features["image_id"],
                'iscrowd' : features["iscrowd"] ,
            }
            if self.use_kps :
                ann['src_num_keypoints'] = tf.cast( tf.sparse.to_dense(features['num_keypoints']) , dtype=tf.int32)
                ann['src_keypoints'] = data['kps']
                
            data['meta_info'] = ann
        return data   
