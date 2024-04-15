

from typing import Any, Callable, Dict, List, Optional, Tuple, Union,Sequence
import tensorflow as tf
from lib.Registers import MODELS
from lib.models.model_utils import make_divisible, make_round
from lib.models.backbones.base_backbone import BaseBackbone
from lib.layers import SPPF, Conv2D_BN
from lib.models.modules import CSPLayerWithTwoConv


BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.97
#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
@MODELS.register_module()
class  YOLOv8CSPDarknet(BaseBackbone):
    
    VERSION = '1.0.0'
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    # the final out_channels will be set according to the param.
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, None, 3, True, True]],   
    }

    r"""Implements the YOLOV8 backbone for object detection

     This backbone is a variant of the `CSPDarkNetBackbone` architecture.
 
                deepen_factor   widen_factor    max_channels       csp_channels       csp_depth
    -------------------------------------------------------------------------------------------------------------
    YOLOv8-x       1.             1.25          512                [160, 320, 640, 640]  [3, 6, 6, 3]
    YOLOv8-l       1.             1.            512                [128, 256, 512, 512]  [3, 6, 6, 3]
    YOLOv8-m       0.67           0.75          768                [96, 192, 384, 576]   [2, 4, 4, 2]
    YOLOv8-s       0.33           0.5           1024               [64, 128, 256, 512]   [1, 2, 2, 1]
    YOLOv8-n       0.33           0.25          1024               [32, 64, 128, 256]    [1, 2, 2, 1]
    
    Args:
        model_input_shape (Tuple[int,int]) : default to (640,640)
        arch_type(str) :   Architecture of CSP-Darknet, from {P5}. Defaults to 'P5'
        last_stage_out_channels (int) :  Final layer output channel, Defaults to 1024.
        deepen_factor (float) : Depth multiplier, multiply number of blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float) :  Width multiplier, multiply number of channels in each layer by this amount. Defaults to 1.0.
        expand_ratio (float) :  Ratio to adjust the number of hidden channels in CSP layers. Defaults to 0.5.
        use_depthwise (bool) :  Whether to use depthwise separable convolution in bottleneck blocks. Defaults to False.
        out_feat_indices (List[int]) : Output from which stages,  default to None 
                out_feat_indices=[3,4,5] that means model outputs [P3,P4,P5], here P2 means 1/4 scale of input image.
                Support negative indices, i.e. [-3,-2,-1] means [P3,P4,P5]
                it can be noted that the order of feats must be from larger scale to small , i.e. P3->P5 , and P5->P3 is invalid
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in DepthwiseConvModule and ConvModule, defaults to 'silu'.
        data_preprocessor (dict) = default to None
        depoly (bool): determine depolyment config for each cell . default to None, 
                depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfgs depend on above args.
                depoly = True => to use deployment config, only conv layer will be bulit
                depoly = False => to use training config() ,
    References:
            - [Based on implementation of 'YOLOv8CSPDarknet' @mmdet] (https://github.com/open-mmlab/mmyolo/blob/main/mmyolo/models/backbones/csp_darknet.py)
            - [config of 'YOLOv8CSPDarknet' @mmdet] (https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py)
            - [Inspired on 'YOLOV8Backbone' @leondgarse] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/yolov8/yolov8.py)
            - [Inspired on 'YOLOV8Backbone' @keras-cv] (https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/object_detection/yolo_v8/yolo_v8_backbone.py)

    Note :
       - 
       - 
       - 
       - 
    Example:
        '''Python
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        'CSPNeXt-m'
        model = CSPNeXt(model_input_shape=(640,640),
                        arch_type  = 'P5',
                        deepen_factor = 0.67,
                        widen_factor = 0.75,
                        expand_ratio= 0.5,
                        use_channel_attn = True,
                        use_depthwise = False,
                        out_feat_indices = [3,4,5],
                        bn_epsilon = 1e-5,
                        bn_momentum = 0.9,
                        activation = 'silu',
                        data_preprocessor= None,
                        deploy = None)

        model.summary(200)

    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
   # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    # the final out_channels will be set according to the param.

    def __init__(self,      
        model_input_shape : Tuple[int,int]=(640,640),
        arch_type: str = 'P5',
        last_stage_out_channels: int = 1024,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        expand_ratio: float = 0.5,
        use_depthwise: bool = False,
        out_feat_indices : Optional[List[int]] = None,
        data_preprocessor: dict = None,
        name =  'YOLOv8CSPDarknet',
        *args, **kwargs) -> None:

        self.arch_settings[arch_type][-1][1] = last_stage_out_channels
        self.arch_setting = self.arch_settings[arch_type]

        self.last_stage_out_channels = last_stage_out_channels
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor

        self.expand_ratio = expand_ratio
        self.use_depthwise = use_depthwise

  
        'basic config'
        self.out_feat_indices = out_feat_indices
        super().__init__(input_size =  (*model_input_shape,3),
                        data_preprocessor = data_preprocessor,
                        name = name, **kwargs)

    def call(self,  x:tf.Tensor)->tf.Tensor:
        
        extracted_feat_indices = [] if self.out_feat_indices is None else self.out_feat_indices
        if extracted_feat_indices != [] : 
            extracted_feats = [] 

        'stem'
        stem_channels = make_divisible(self.arch_setting[0][0], self.widen_factor)
        x = Conv2D_BN(filters = stem_channels,
                    kernel_size=3,
                    strides=2,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy  = self.deploy,
                    name = 'Stem_P1_Conv1')(x)
          
         
        for i, (in_channels, out_channels, num_bottleneck_block, add_identity, use_spp) in enumerate(self.arch_setting):
            'cfg of ith_stage'
            p_feat = f'Stage_P{i+2}'
            out_channels = make_divisible(out_channels, self.widen_factor)
            num_bottleneck_block = make_round(num_bottleneck_block, self.deepen_factor)
  
            x = Conv2D_BN(filters = out_channels,
                    kernel_size=3,
                    strides=2,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy  = self.deploy,
                    name = f'{p_feat}_downsample')(x)

    
            x = CSPLayerWithTwoConv(out_channels = out_channels,
                                expand_ratio= self.expand_ratio,
                                kernel_sizes = [3,3],
                                csp_depthes = num_bottleneck_block,
                                use_shortcut = add_identity,
                                use_depthwise  = self.use_depthwise,
                                bn_epsilon = self.bn_epsilon,
                                bn_momentum = self.bn_momentum,
                                activation = self.act_name,
                                deploy = self.deploy,
                                name = f'{p_feat}_c2f')(x)  
             
            if use_spp : 
                x = SPPF(out_channels= out_channels,
                        pool_size =5,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        activation = self.act_name,
                        name=f'{p_feat}_SPPF')(x)
                
            if (i+2 in extracted_feat_indices) or ((i+2)-6 in extracted_feat_indices):
                extracted_feats.append(x)

        return  x if extracted_feat_indices==[] else extracted_feats
    
@MODELS.register_module()
def YOLOv8_N(model_input_shape=(640,640), 
    arch_type  = 'P5',
    data_preprocessor= None,  
    deploy = False, 
    **kwargs):
    last_stage_out_channels = 1024
    deepen_factor = 0.33
    widen_factor = 0.25
    bn_epsilon = BATCH_NORM_EPSILON
    bn_momentum = BATCH_NORM_MOMENTUM
    activation = 'swish'
    return YOLOv8CSPDarknet(**locals(), name='YOLOv8_n', **kwargs)

@MODELS.register_module()
def YOLOv8_S(model_input_shape=(640,640), 
    arch_type  = 'P5',
    data_preprocessor= None,  
    deploy = False, 
    **kwargs):
    last_stage_out_channels = 1024
    deepen_factor = 0.33
    widen_factor = 0.5
    bn_epsilon = BATCH_NORM_EPSILON
    bn_momentum = BATCH_NORM_MOMENTUM
    activation = 'swish'
    return YOLOv8CSPDarknet(**locals(), name='YOLOv8_s', **kwargs)

@MODELS.register_module()
def YOLOv8_M(model_input_shape=(640,640), 
    arch_type  = 'P5',
    data_preprocessor= None,  
    deploy = False, 
    **kwargs):
    last_stage_out_channels = 768
    deepen_factor = 0.67
    widen_factor = 0.75
    bn_epsilon = BATCH_NORM_EPSILON
    bn_momentum = BATCH_NORM_MOMENTUM
    activation = 'swish'
    return YOLOv8CSPDarknet(**locals(), name='YOLOv8_m', **kwargs)

@MODELS.register_module()
def YOLOv8_L(model_input_shape=(640,640), 
    arch_type  = 'P5',
    data_preprocessor= None,  
    deploy = False, 
    **kwargs):
    last_stage_out_channels = 512
    deepen_factor = 1.
    widen_factor = 1.
    bn_epsilon = BATCH_NORM_EPSILON
    bn_momentum = BATCH_NORM_MOMENTUM
    activation = 'swish'
    return YOLOv8CSPDarknet(**locals(), name='YOLOv8_l', **kwargs)

@MODELS.register_module()
def YOLOv8_X(model_input_shape=(640,640), 
    arch_type  = 'P5',
    data_preprocessor= None,  
    deploy = False, 
    **kwargs):
    last_stage_out_channels = 512
    deepen_factor = 1.
    widen_factor = 1.25
    bn_epsilon = BATCH_NORM_EPSILON
    bn_momentum = BATCH_NORM_MOMENTUM
    activation = 'swish'
    return YOLOv8CSPDarknet(**locals(), name='YOLOv8_x', **kwargs)


#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
