import tensorflow as tf
from tensorflow import Tensor
from typing import Dict, List, Optional, Tuple, Union
from lib.Registers import TRANSFORMS
from  .base import CVBaseTransformLayer
#########################################################################
#
#
############################################################################
@TRANSFORMS.register_module()
class RandomFlip(CVBaseTransformLayer):
    version = '2.0.0'

    r"""Randomly flip the image, bbox and keypoints.

    date : 2024/3/24
    author : Dr. Shen 

    support multi bboxes in one image

    Required Keys:
        - img
        - img_shape
        - bbox (optional)
        - kps (optional) 
    Required Data dtype:
        - tf.Tensor 

    Args:
        prob (float): The flipping probability. Defaults to 0.5

    TO DO: support batch image
    """
    def __init__(
        self, 
        prob : Optional[float] = 0.5,
        kps_index_flip : List[int] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],  
        labels_flip : Optional[Dict]= None ,       
        **kwargs
    ): 
        super().__init__(**kwargs)
        
        self.kps_index_flip = kps_index_flip
        self.prob = prob
        if labels_flip is not None :
            ''' 
            i.e . labels_flip = {1:1, 2:2, 3:4, 4:3}
            here, label 3 and 4 will become as label 4 and 3, respectiely, after image flipping
            and label 1 and 2 are no change
            
            '''
            table = tf.lookup.StaticVocabularyTable(
                tf.lookup.KeyValueTensorInitializer(
                        list(labels_flip.keys()),
                        list(labels_flip.values()),
                        key_dtype=tf.int64,
                        value_dtype=tf.int64,
                ),
                num_oov_buckets=1,
            )
            self.swap_labels = lambda x : tf.cast( table.lookup(tf.cast(x, tf.int64)), dtype=x.dtype)


    def transform(
            self, data : Dict[str,Tensor], **kwargs) -> Dict[str,Tensor]:
        """The transform function of :class:`RandomFlip`.
        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            data (dict):  The result dict
            i.e. data = {"image": img_tensor, "bbox":bbox_tensor, "kps": kps_tensor}
        Returns:
            dict: The output result dict. like input data
        """
        #assert data["image"].shape.rank==3, 'not yet support batched transform'
        # if  self.prob < tf.random.uniform(()):
        #     return data  
        bool_op = self._rand_bool(prob=(self.prob))
        
        'formatting image type'
        image = self.img_to_tensor(data["image"]) 
        
        'img flip'
        image = tf.cond(
            bool_op, 
            lambda : tf.image.flip_left_right(image),
            lambda : image
        )
        #image = tf.image.flip_left_right(image) #tf.tensor

        'update img and modify its type to data'
        data = self.update_data_img(image, data)  #tf.tensor->tf.ragged_tensor or tf.tensor->tf.tensor

        img_w = tf.cast(tf.shape(image)[1], dtype=self.compute_dtype)
        'bbox flip ; support multi bboxes i.e. bboxes.shape=(num_bbox,4) or (4,) '
        if data.get('bbox', None) is not None and bool_op:
            bbox = data["bbox"]
            data["bbox"] = tf.stack( 
                [ img_w-bbox[...,0]-bbox[...,2], bbox[...,1], bbox[...,2], bbox[...,3]], axis=-1
            )

        'kps flip'
        if data.get('kps', None) is not None  and bool_op:
            kps = data["kps"]
            cond = tf.expand_dims(tf.equal(kps[...,2], 0.),axis=-1)
            kps = tf.stack([ img_w - kps[...,0], kps[...,1], kps[...,2]],axis=-1)
            kps = tf.where(cond, tf.zeros_like(0., dtype=kps.dtype), kps) 
            # 'flip_kps'
            data["kps"]  = tf.gather(kps, self.kps_index_flip, axis=-2)

        'lables flip if it is necessary '
        if data.get('labels', None) is not None and hasattr(self,'swap_labels'):
            data["labels"]  = tf.cond(
                bool_op, 
                lambda :  self.swap_labels( data["labels"] ),
                lambda :  data["labels"]
            )

        return data