
import tensorflow as tf
from typing import Dict
from lib.datasets.transforms.utils import rand_prob
from lib.Registers import TRANSFORMS
from .base import CVBaseTransformLayer
@TRANSFORMS.register_module()
class RandomMixUp(CVBaseTransformLayer):
    VERSION = '2.0.0'
    r""" MixUp implements the MixUp data augmentation technique 
    date : 2024/3/26
    author : Dr. David Shen 

    TO DO : need to test classification task
        

    The mixup transform steps are as follows:
        1. Another random image is picked from batched_dataset and generate permutation order
        2. The target images generated by mixup transform is the weighted average of mixup
           image and origin image.
        3. gather corresponding lebels,bboxes and kps by permutation order 
        4. if task is object or pose detection, mixup transform  concat labels (bboxes / kps) of mixup and origin image 
           if task is classification (no bboxes exsit and lables is one hot format), mixup transform output weighted lables

    Required Keys:
        - image ----- TensorSpec(shape=(b, h, w, 3), dtype=tf.uint8, name=None)
        - labels ----- RaggedTensorSpec(TensorShape([b, None]), tf.int32, 1, tf.int64) 
        - bbox(optional) ----- RaggedTensorSpec(TensorShape([b, None, 4]), tf.float32, 1, tf.int64)
        - kps(optional) ----- RaggedTensorSpec(TensorShape([b, None, num_joints, 3]), tf.float32, 1, tf.int64)

    Modified Keys:
        - image 
        - labels
        - bbox(optional)
        - kps(optional)

    Args:
        alpha (float): Float between 0 and 1. Inverse scale parameter for the gamma
            distribution. This controls the shape of the distribution from which
            the smoothing values are sampled. Defaults to 32, which is a
            recommended value from yolo when training object detection model.
            if training imagenet1k classification model, recommended value is 0.2
        prob (float) : The probability to enable MixUp . Defaults to 0.5

    References:
        - [MixUp paper] (https://arxiv.org/abs/1710.09412).
        - [MixUp for Object Detection paper] (https://arxiv.org/pdf/1902.04103).
        - [MixUp implement of YOLO] (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py).
        - [MixUp implement of keras_cv] (https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing/mix_up.py).
        - [MixUp implement of mmdet] ( https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/transforms/transforms.py).

    Sample usage:

    ```python
    import copy
    from lib.Registers import DATASETS
    from lib.datasets.transforms import Mosaic, MixUp
    from lib.datasets.transforms.utils import PackInputTensorTypeSpec
    ---------------------pre_transform------------------------------------------  
    tfrec_val_dataset = dict(
        type = 'Test_CocoStylePoseDataset_Parser', 
        data_root = val_tfrec_bottmup_pose_dir,
        is_whole_body_boxes = False,
        sel_kpt_idx  = [i for i in range(17)],
    )

    mosaic = Mosaic(
        target_size = (640,640),
        center_ratio_range = (0.75, 1.25),
        scaling_ratio_range= (0.75, 1.25),
        num_used_kps = 17,
        prob = 1.
    )
    ---------------------------------------------------------------  
    mixup = MixUp(
        alpha = 0., 
        prob = 1.
    )

    test_dataloader_cfg =  dict(
        type = 'dataloader',
        batch_size = 16,
        prefetch_size = 4,
        shuffle  =  True,
        tfrec_datasets_list = [tfrec_val_dataset],
        augmenters = [mosaic, mixup], 
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
    ------------------------Review Results-------------------------------------- 
    tfds_builder.get_pipeline_cfg() 
    batch_id = 0
    for features in batch_dataset.take(1):
    
        PackInputTensorTypeSpec(features,{}, show_log=True)
        image = features["image"][batch_id].numpy()
        labels = features['labels'][batch_id]
        bboxes = features['bbox'][batch_id]
        
        if features.get('kps',None) is not None :
            kpts = features['kps'][batch_id]
            print(f"labels : {labels.shape}, bboxes : {bboxes.shape}, kpts : {kpts.shape}")
        
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

    """
    def __init__(
        self,
        alpha : float = 32.,
        prob: float = 0.,
        **kwargs
    ):
        super().__init__(**kwargs)
        'basic cfg'
        self.prob = prob
        self.alpha = alpha


    @tf.function
    def call(
        self, dic : Dict, *args, **kwargs
    ) ->Dict:
        
        data = {k:v for k,v in dic.items()}
        del dic
        if not isinstance(data,dict):
            raise TypeError(
                "data type must be dict[str,Tensor],"
                f" but got {type(data)}@{self.__class__.__name__}"
            )
        
        if data['image'].shape.rank == 4:
            if not isinstance(data['image'],tf.Tensor):
                raise TypeError(
                    "the type of data['image'] must be tf.Tensor, "
                    f"but got {type(data['image'])} @{self.__class__.__name__}"
                ) 
            data = self.batched_transform(data)
        else:
            raise ValueError(
                "MixUp must receive batched images, "
                f"but got {data['image'].shape.rank} @{self.__class__.__name__}"
                "Please call this transform by sending samples with batch_size=4 at least"
            )
        return data
    
    def _sample_from_beta(
            self, alpha, beta, shape
    ):
        if alpha == 0. :
            return tf.ones(shape, dtype=tf.float32)*0.5
        
        sample_alpha = tf.random.gamma(
            shape,
            alpha=alpha,
        )
        sample_beta = tf.random.gamma(
            shape,
            alpha=beta,
        )
        return sample_alpha / (sample_alpha + sample_beta)
    


    def batched_transform(
        self,data : Dict[str,tf.Tensor]
    )->Dict[str,tf.Tensor]:
        
        if self.prob < rand_prob():
            return data
        
        images = data["image"]
        img_dtype = images.dtype
        batch_size=tf.shape(images)[0]

        'batch_idx for selecting images'
        permutation_order = tf.random.shuffle(
            tf.range(0, batch_size)
        )
        
        lambda_sample = self._sample_from_beta(
            self.alpha, self.alpha, (batch_size,)
        )
        #lambda_sample = tf.ones(shape=(batch_size,), dtype=tf.float32)*0.5

        lambda_sample = tf.cast(
            tf.reshape(lambda_sample, [-1, 1, 1, 1]), dtype=self.compute_dtype
        )

        'update images'
        images =  tf.cast(images, dtype=self.compute_dtype)
        mixup_images = tf.gather(images, permutation_order)
        #images = lambda_sample * images + (1.0 - lambda_sample) * mixup_images
        data["image"] = tf.cast(
            lambda_sample * images + (1.0 - lambda_sample) * mixup_images, 
            dtype=img_dtype
        )

   
        'upadte onehot labels (tf.tensor) for classification'
        if isinstance(data['labels'], tf.Tensor):
            # need to test
            mixup_labels = tf.gather(data['labels'], permutation_order)
            lambda_sample = tf.reshape(lambda_sample, [-1, 1])
            data['labels'] = (
                lambda_sample * data['labels'] + (1.0 - lambda_sample) * mixup_labels
            )
            return data


        'update labels(tf.RaggedTensor) and bbox(tf.RaggedTensor) for object detection'
        if data.get('bbox',None) is not None:
            mixup_bboxes = tf.gather(data['bbox'], permutation_order)
            data['bbox'] = tf.concat(
                [data['bbox'], mixup_bboxes], axis=-2
            )
            mixup_labels = tf.gather(data['labels'], permutation_order)
            data['labels'] = tf.concat(
                [data['labels'], mixup_labels], axis=-1
            )
            
        'update keypoint for pose detection'
        if data.get('kps',None) is not None:
            mixup_kps = tf.gather(data['kps'], permutation_order) 
            data['kps'] = tf.concat(
                [data['kps'], mixup_kps], axis=-3
            )
            
        return data
    