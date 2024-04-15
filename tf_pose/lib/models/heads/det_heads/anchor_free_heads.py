
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from lib.Registers import MODELS
from lib.models.modules import BaseModule
from lib.layers import Conv2D_BN, DenseFuseLayer
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Activation, Concatenate, Reshape
import math

#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
@MODELS.register_module()
class YOLO_AnchorFreeHead(BaseModule):
    VERSION = '1.1.0'
    r""" 

                                                          
                P3P4P5_in_channels   box_channels   cls_channels   box_head_out   cls_head_out
    ------------------------------------------------------------------------------------------
    YOLOv8-x    [320, 640, 640]          80          
    YOLOv8-l    [256, 512, 512]          64
    YOLOv8-m    [192, 384, 576]          64
    YOLOv8-s    [128, 256, 512]          64
    YOLOv8-n    [ 64, 128, 256]          64

    
    Args:
        reg_max (int) :  in_channels of dense fuse layer(DFL). , 
            Max value of integral set :math: ``{0, ..., reg_max-1}, Defaults to 16.
        num_classes (int) : Number of categories excluding the background
            category. default to 80.
        with_auxiliary_regression_channels (bool) : whether to use auxiliary_regression_channels for training.
            Defaults to False,  when depoly = True, it is invalid and will be always False
            if depoly = False or None, this arg is valid 
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in DepthwiseConvModule and ConvModule, defaults to 'silu'.
        depoly (bool): determine depolyment config for each cell . default to None, 
                depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfgs depend on above args.
                depoly = True => to use deployment config, only conv layer will be bulit
                depoly = False => to use training config() 


    Note :
       -  bbox_outputs : left / top / right / bottom predictions with respect to anchor
       -  with_auxiliary_regression_channels = True :  bbox_dist_outputs.shape =(b, 8400, 4+4*reg_max)
       -  with_auxiliary_regression_channels = False :  bbox_dist_outputs.shape =(b, 8400, 4)
       - 
    References:

            - [Based on implementation of 'apply_yolo_v8_head' @keras-cv ]
                     (https://github.com/keras-team/keras-cv/blob/832e2d9539fd29a8348f38432673d14c18981df2/keras_cv/models/object_detection/yolo_v8/yolo_v8_detector.py#L189)
            - [Inspired on implementation of 'Detect' @ultralytics] 
                    (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py)   
             - [Inspired on implementation of 'YOLOv8HeadModule' @mmyolo] 
                    (https://github.com/open-mmlab/mmyolo/blob/8c4d9dc503dc8e327bec8147e8dc97124052f693/mmyolo/models/dense_heads/yolov8_head.py)              
            - [Inspired on implementation of 'yolov8_head' @leondgarse] 
                    (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/yolov8/yolov8.py#L232)

    Example:
        -------------------------------------------------------
        '''Python
        from lib.models.heads import YOLO_AnchorFreeHead
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input

        head = YOLO_AnchorFreeHead(
                    reg_max= 16, 
                    num_classes=80,
                    activation = 'swish',
                    bn_epsilon = 1e-5,
                    bn_momentum = 0.9,
                    deploy =  False,
                    scope_marks = '_',
                    wrapped_model = False,
                    name='YOLOv8_Head'
        )
        
        feat_p3 = Input((80,80,64))
        feat_p4 = Input((40,40,128))
        feat_p5 = Input((20,20,256))
        features = [feat_p3,feat_p4,feat_p5]
        out = head(features)
        model = Model(features, out)
        model.summary(200, expand_nested=True)

        -------------------------------------------------------
        '''Python
        from lib.models import HEADS

        head_cfg = dict(
                type ='YOLO_AnchorFreeHead',
                reg_max= 16, 
                num_classes=80,
                activation = 'swish',
                bn_epsilon = 1e-5,
                bn_momentum = 0.9,
                deploy =  False,
                scope_marks = '_',
                wrapped_model = False,
                name='YOLOv8_Head'
        )
        head = HEADS.build(head_cfg)

        feat_p3 = Input((80,80,64))
        feat_p4 = Input((40,40,128))
        feat_p5 = Input((20,20,256))
        features = [feat_p3,feat_p4,feat_p5]
        out = head(features)
        model = Model(features, out)
        model.summary(200, expand_nested=True)

    """                    
    def __init__(self,
            reg_max: int = 16, 
            num_classes : int=80,
            activation : str='swish',
            with_auxiliary_regression_channels : bool = False,
            bbox_conv_groups : int = 1,
            name=None, 
            **kwargs):
        super().__init__(name=name, 
                        activation=activation, 
                        **kwargs)
     
        self.reg_max = reg_max
        self.box_regression_channels = self.reg_max*4
        self.num_classes = num_classes
        self.bbox_conv_groups = bbox_conv_groups

        self.with_auxiliary_regression = with_auxiliary_regression_channels

    def build(self, input_shape):
        ''' 
        input_shape : [P3:(),P4:()]
        
        '''
        self.box_channels = max(self.reg_max*4, input_shape[0][-1]//4)
        self.class_channels = max(min(self.num_classes*2,128), input_shape[0][-1])

        self.cls_pwconv_bias_init = []
        for idx in range(len(input_shape)):
            strides = 2 ** (idx + 3)
            bias_init = tf.constant_initializer(
                        math.log(5 / self.num_classes / (640 /strides) ** 2)
                    )
            self.cls_pwconv_bias_init.append(bias_init)

        
        
    def  call(self, features):
        outputs  = []

        # bbox_preds = []
        # class_preds = []
        for idx, feature in enumerate(features):
            cur_name = self.name + f"P{idx+3}"
            #----------------bbox----------------
            bbox_pred = Conv2D_BN(
                filters = self.box_channels,
                kernel_size=3,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                activation = self.act_name,
                deploy= self.deploy,
                name=f'{cur_name}_box_conv1'
            )(feature) 

            bbox_pred = Conv2D_BN(
                filters = self.box_channels,
                kernel_size=3,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                activation = self.act_name,
                groups = self.bbox_conv_groups,
                deploy= self.deploy,
                name=f'{cur_name}_box_conv2'
            )(bbox_pred) 
            
            bbox_pred = Conv2D(
                filters=self.reg_max*4, 
                kernel_size=1, 
                use_bias=True,
                groups = self.bbox_conv_groups,
                bias_initializer="ones",
                name=f"{cur_name}_box_pwconv",
                dtype=tf.float32
            )(bbox_pred)
     
            #---------------class-----------------
            class_pred = Conv2D_BN(
                filters = self.class_channels,
                kernel_size=3,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                activation = self.act_name,
                deploy= self.deploy,
                name=f'{cur_name}_cls_conv1'
            )(feature) 

            class_pred = Conv2D_BN(
                filters = self.class_channels,
                kernel_size=3,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                activation = self.act_name,
                deploy= self.deploy,
                name=f'{cur_name}_cls_conv2'
            )(class_pred) 
            
            # class_pred = Conv2D_BN(
            #     filters = self.num_classes,
            #     kernel_size=1,
            #     use_bn = False,
            #     use_bias=True,
            #     activation = 'sigmoid',
            #     name=f"{cur_name}_cls_pwconv"
            # )(class_pred) 

            class_pred = Conv2D(
                filters=self.num_classes, 
                kernel_size=1, 
                use_bias=True,
                bias_initializer=self.cls_pwconv_bias_init[idx],
                name=f"{cur_name}_cls_pwconv",
                dtype=tf.float32,
            )(class_pred)


            out = Concatenate(
                axis=-1, name=f"{cur_name}_concat"
            )([bbox_pred, class_pred])

            out = Reshape([-1,out.shape[-1]], 
                          name=f"{cur_name}_reshape"
            )(out)
            outputs.append(out)

        outputs = Concatenate(
            axis=1, name=self.name + "concat"
        )(outputs)

        bbox_reg_outputs = outputs[:,:,:self.box_regression_channels]
        class_outputs = outputs[:,:,self.box_regression_channels:]

        'final outputs'
        # if self.with_auxiliary_regression = True : 
        # bbox_dist_outputs.shape =(b, 8400, 4+4*reg_max)
        #    

        bbox_dist_outputs = DenseFuseLayer(
            reg_max=self.reg_max,  
            name=self.name + "box_dfl",
            deploy=self.deploy,
            with_regression_channels = self.with_auxiliary_regression,
            dtype=tf.float32, 
        )(bbox_reg_outputs)

        bbox_dist_outputs = Activation(
            "linear", dtype="float32", name=self.name +"box_output"
        )(bbox_dist_outputs)

        class_outputs = Activation(
            'sigmoid',  dtype=tf.float32, name=self.name +"cls_sigmoid"
        )(class_outputs)


        class_outputs = Activation(
            "linear", dtype="float32", name=self.name +"class_output"
        )(class_outputs)

        #if self.use_regression_auxiliary_head:
        return bbox_dist_outputs, class_outputs
    

#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
@MODELS.register_module()
class YOLO_AnchorFreePoseHead(YOLO_AnchorFreeHead):
    def __init__(
            self, kpt_shape : Tuple[int] = (17,3), **kwargs
    ):
        super().__init__(**kwargs)
        self.det_head = YOLO_AnchorFreeHead.call
        self.kpt_shape = kpt_shape
    

    def build(self, input_shape):
        ''' 
        input_shape : [P3:(),P4:()]
        
        '''
        super().build(input_shape)
        self.kpts_channels = max(
            self.kpt_shape[0]*self.kpt_shape[1], input_shape[0][-1]//4
        )
        
        
    def call(self, features):

        bbox_dist_outputs, class_outputs = self.det_head(self,features)
        outputs  = []
        for idx, feature in enumerate(features):
            cur_name = self.name + f"P{idx+3}"
            #----------------bbox----------------
            kpts_pred = Conv2D_BN(
                filters = self.kpts_channels,
                kernel_size=3,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                activation = self.act_name,
                deploy= self.deploy,
                name=f'{cur_name}_kpt_conv1'
            )(feature) 

            kpts_pred = Conv2D_BN(
                filters = self.kpts_channels,
                kernel_size=3,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                activation = self.act_name,
                deploy= self.deploy,
                name=f'{cur_name}_kpt_conv2'
            )(kpts_pred) 
            
            kpts_pred = Conv2D(
                filters= self.kpt_shape[0]* self.kpt_shape[1], 
                kernel_size=1, 
                use_bias=True,
                name=f"{cur_name}_kpt_pwconv",
                dtype=tf.float32
            )(kpts_pred) 
            kpts_pred = Reshape(
                [-1, self.kpt_shape[0], self.kpt_shape[1]], 
                name=f"{cur_name}_kpt_reshape"
            )(kpts_pred)
            outputs.append(kpts_pred)

        kpt_outputs = Concatenate(
            axis=1, name=self.name + "kpt_feat_concat"
        )(outputs)

        obj_kpt = Activation(
                'sigmoid', 
                name=self.name +"kpt_sigmoid",
                dtype=tf.float32,
        )( kpt_outputs[...,2:3])

        kpt_outputs = Concatenate(
            axis=-1, name=self.name + "kpt_dim_concat"
        )([ kpt_outputs[...,:2], obj_kpt])


        kpt_outputs = Activation(
            "linear", dtype="float32", name="kpt"
        )(kpt_outputs)

        return bbox_dist_outputs, class_outputs, kpt_outputs
    