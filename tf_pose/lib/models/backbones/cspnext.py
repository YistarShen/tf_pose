'tf layers'
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tensorflow import Tensor
import tensorflow as tf
from .base_backbone import BaseBackbone
from lib.layers import Conv2D_BN
from lib.models.modules import CrossStagePartial
from lib.layers import SPPF
from lib.Registers import MODELS


BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.97
#--------------------------------------------------------------------------------------------------------------
#
#--------------------------------------------------------------------------------------------------------------
@MODELS.register_module()
class CSPNeXt(BaseBackbone):
    
    VERSION = '1.0.0'
    r""" CSPNeXt, CSPNeXt backbone used in RTMDet/RTMPose

                deepen_factor        widen_factor
    ---------------------------------------------
    CSPNeXt-x       1.33                  1.25
    CSPNeXt-l       1.                    1.
    CSPNeXt-m       0.67                  0.75
    CSPNeXt-s       0.33                  0.5
    CSPNeXt-t       0.167                 0.375
    
    Args:
        model_input_shape (Tuple[int,int]) : default to (640,640)
        arch_type(str) :  Architecture of CSPNeXt, from {P5, P6}. Defaults to 'P5'
        deepen_factor (float) : Depth multiplier, multiply number of blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float) :  Width multiplier, multiply number of channels in each layer by this amount. Defaults to 1.0.
        expand_ratio (float) :  Ratio to adjust the number of hidden channels in CSP layers. Defaults to 0.5.
        use_channel_attn (bool) : Whether to add channel attention in each stage, default to True
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
            - [Based on implementation of CSPNeXt @mmdet] (https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/cspnext.py)
            - [config of CSPNeXt @mmdet] (https://github.com/open-mmlab/mmdetection/tree/main/mmdet/configs/rtmdet)

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
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }
    def __init__(self,      
        model_input_shape : Tuple[int,int]=(256,192),
        arch_type: str = 'P5',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        expand_ratio: float = 0.5,
        use_channel_attn: bool = True,
        use_depthwise: bool = False,
        out_feat_indices : Optional[List[int]] = None,
        data_preprocessor: dict = None,
        name =  'cspnext',
        *args, **kwargs) -> None:

        self.arch_setting = self.arch_settings[arch_type]
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor


        'elan config'
        self.expand_ratio = expand_ratio
        self.use_channel_attn = use_channel_attn
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
        stem_channels = self.arch_setting[0][0]*self.widen_factor

        x = Conv2D_BN(filters = stem_channels//2,
                    kernel_size=3,
                    strides=2,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy  = self.deploy,
                    name = 'Stem_P1_Conv1')(x)
        
        x = Conv2D_BN(filters = stem_channels//2,
                    kernel_size=3,
                    strides=1,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy  = self.deploy,
                    name = 'Stem_P1_Conv2')(x)

        x = Conv2D_BN(filters = stem_channels,
                    kernel_size=3,
                    strides=1,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy  = self.deploy,
                    name = 'Stem_P1_Conv3')(x)   
         
        for i, (in_channels, out_channels, num_bottleneck_block, add_identity, use_spp) in enumerate(self.arch_setting):
            'cfg of ith_stage'
            p_feat = f'Stage_P{i+2}'
            out_channels =  out_channels*self.widen_factor
            num_bottleneck_block = int( max(num_bottleneck_block*self.deepen_factor,1) )
  
            x = Conv2D_BN(filters = out_channels,
                    kernel_size=3,
                    strides=2,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy  = self.deploy,
                    name = f'{p_feat}_Conv1')(x)
            
            if use_spp : 
                x = SPPF(out_channels= out_channels,
                        pool_size =5,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        activation = self.act_name,
                        name=f'{p_feat}_SPPF')(x)
        
            x = CrossStagePartial(out_channels = out_channels,
                                expand_ratio= self.expand_ratio,
                                kernel_sizes = [3,5],
                                csp_type = 'C3',
                                csp_depthes = num_bottleneck_block,
                                use_shortcut = add_identity,
                                use_cspnext_block = True,
                                use_depthwise  = self.use_depthwise,
                                use_channel_attn = self.use_channel_attn,
                                bn_epsilon = self.bn_epsilon,
                                bn_momentum = self.bn_momentum,
                                activation = self.act_name,
                                deploy = self.deploy,
                                name = f'{p_feat}_CSP')(x)  
      
            if (i+2 in extracted_feat_indices) or ((i+2)-6 in extracted_feat_indices):
                extracted_feats.append(x)

        return  x if extracted_feat_indices==[] else extracted_feats
    
    
"""
                deepen_factor        widen_factor
    ---------------------------------------------
    CSPNeXt-x       1.33                  1.25
    CSPNeXt-l       1.                    1.
    CSPNeXt-m       0.67                  0.75
    CSPNeXt-s       0.33                  0.5
    CSPNeXt-t       0.167                 0.375
"""
#def CSPNeXt_L(input_shape=(640,650,3))


@MODELS.register_module()
def CSPNeXt_L(model_input_shape=(640,640), 
    arch_type  = 'P5',
    data_preprocessor= None,  
    deploy = False, 
    **kwargs):
    deepen_factor = 1.0
    widen_factor = 1.0
    bn_epsilon = BATCH_NORM_EPSILON
    bn_momentum = BATCH_NORM_MOMENTUM
    activation = 'silu'
    return CSPNeXt(**locals(), name='CSPNeXt_L', **kwargs)


@MODELS.register_module()
def CSPNeXt_M(model_input_shape=(640,640), 
    arch_type  = 'P5',
    data_preprocessor= None,  
    deploy = False, 
    **kwargs):
    deepen_factor = 0.67
    widen_factor = 0.75
    bn_epsilon = BATCH_NORM_EPSILON
    bn_momentum = BATCH_NORM_MOMENTUM
    activation = 'silu'
    return CSPNeXt(**locals(), name='CSPNeXt_M', **kwargs)



@MODELS.register_module()
def CSPNeXt_S(model_input_shape=(640,640), 
    arch_type  = 'P5',
    data_preprocessor= None,  
    deploy = False, 
    **kwargs):
    deepen_factor = 0.33
    widen_factor = 0.5
    bn_epsilon = BATCH_NORM_EPSILON
    bn_momentum = BATCH_NORM_MOMENTUM
    activation = 'silu'
    return CSPNeXt(**locals(), name='CSPNeXt_S', **kwargs)



@MODELS.register_module()
def CSPNeXt_Tiny(model_input_shape=(640,640), 
    arch_type  = 'P5',
    data_preprocessor= None,  
    deploy = False, 
    **kwargs):
    deepen_factor = 0.167
    widen_factor = 0.375
    bn_epsilon = BATCH_NORM_EPSILON
    bn_momentum = BATCH_NORM_MOMENTUM
    activation = 'silu'
    return CSPNeXt(**locals(), name='CSPNeXt_Tiny', **kwargs)

    

