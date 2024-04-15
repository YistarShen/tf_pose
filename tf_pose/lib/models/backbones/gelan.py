'tf layers'
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
from tensorflow import Tensor
import tensorflow as tf
from tensorflow.keras.layers import Layer
from .base_backbone import BaseBackbone
from lib.layers import Conv2D_BN
from lib.Registers import MODELS
from lib.layers import SPPELAN, ADown
from lib.models.modules import RepNCSPELAN4
from lib.models.model_utils import make_divisible, make_round

BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.97
#--------------------------------------------------------------------------------------------------------------
#
#--------------------------------------------------------------------------------------------------------------
@MODELS.register_module()
class  GELAN(BaseBackbone):
    
    VERSION = '1.0.0'
    r""" GELAN (Generalized Efficient Layer Aggregation Network) used in YOLOv9
    GELAN = CSPNet + ELAN
    YOLOv9 = GELAN + Auxiliary Reversible Branch 

    GELAN : Main Branch
    Auxiliary Reversible Branch  : used by Programmable Gradient Information(PGI) technique 

    """
    
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    # the final out_channels will be set according to the param.
    # arch_settings = {
    #     'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
    #            [256, 512, 6, True, False], [512, None, 3, True, True]],   
    # }
    # From left to right:
    # in_channels, out_channels, hidden_channels, num_blocks, add_identity, use_spp
    # the final out_channels will be set according to the param.
    # [ in_channels, out_channels, hidden_channels, num_blocks, add_identity, use_ADown]
    # arch_settings = {
    #     'P5': [
    #             [64, 256, 128, 1, True, False], 
    #             [256, 512, 256, 1, True, True],
    #             [512, 512, 512, 1, True, True], 
    #             [512, 512, 512, 1, True, True]
    #         ]  
    # }

    arch_settings = OrderedDict(
        P2 = [64, 256, 128, 1, True, False],
        P3 = [256, 512, 256, 1, True, True],
        P4 = [512, 512, 512, 1, True, True], 
        P5 = [512, 512, 512, 1, True, True]  
    )


    def __init__(self,      
        model_input_shape : Tuple[int,int]=(640,640),
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        expand_ratio: float = 0.5,
        use_depthwise: bool = False,
        use_aux_branch : bool = False,
        out_feat_indices : Optional[List[int]] = None,
        data_preprocessor: dict = None,
        name =  'YOLOv9',
        *args, **kwargs
    ) -> None:

        self.arch_setting = self.arch_settings
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.expand_ratio = expand_ratio
        self.use_depthwise = use_depthwise
        self.use_aux_branch = use_aux_branch
        'basic config'
        self.out_feat_indices = out_feat_indices

        super().__init__(
            input_size =  (*model_input_shape,3),
            data_preprocessor = data_preprocessor,
            name = name, **kwargs
        )
  
    def extract_aux_layers(self):
        'multi_levels_aux_layers'
        return [
            super().get_layer('Silence').output,
            super().get_layer('StageP3_Out').output,
            super().get_layer('StageP4_Out').output,
            super().get_layer('StageP5_Out').output
        ]

    def call(self,  x : tf.Tensor)->tf.Tensor:
        
        extracted_feat_indices = [] if self.out_feat_indices is None else self.out_feat_indices
        if extracted_feat_indices != [] : 
            extracted_feats = [] 

        x = Layer(name='Silence')(x)

        'stem'
        stem_channels = make_divisible(self.arch_setting['P2'][0], self.widen_factor)
        x = Conv2D_BN(
            filters = stem_channels,
            kernel_size=3,
            strides=2,
            bn_epsilon = self.bn_epsilon,
            bn_momentum = self.bn_momentum,
            activation = self.act_name,
            deploy  = None,
            name = 'StemP1_Conv1'
        )(x)
          
        
        for i, (key, vals) in enumerate(self.arch_setting.items()):   
 
            'cfg of ith_stage'
            p_feat = f'Stage{key}'
            in_channels, out_channels, hidden_channels, num_bottleneck_block, add_identity, use_ADown = vals

            out_channels = make_divisible(out_channels, self.widen_factor)
            hidden_channels = make_divisible(hidden_channels, self.widen_factor)
            num_bottleneck_block = make_round(num_bottleneck_block, self.deepen_factor)

            if not use_ADown :
                x = Conv2D_BN(
                        filters = hidden_channels,
                        kernel_size=3,
                        strides=2,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        activation = self.act_name,
                        deploy  = None,
                        name = f'{p_feat}_downsample'
                )(x)
            else:
                x = ADown(
                        out_channels =  hidden_channels,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        activation = self.act_name,
                        name= f'{p_feat}_downsample'
                )(x) 
            
            x = RepNCSPELAN4(
                    out_channels = out_channels,
                    hidden_channels = hidden_channels,
                    csp_depthes  = num_bottleneck_block,
                    csp_exapnd_ratio  = 0.5,
                    kernel_sizes = [3,3],
                    use_shortcut = add_identity,
                    use_depthwise = self.use_depthwise,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy = self.deploy,
                    name  =f'{p_feat}_RepCSPELAN4',  
            )(x) 

            x = Layer(name=f'{p_feat}_Out')(x)
 
            if (i+2 in extracted_feat_indices) or ((i+2)-6 in extracted_feat_indices):
                extracted_feats.append(x)


        #extracted_feats = self._aux_branch([aux_x]+extracted_feats)
        #self._aux_branch([input]+extracted_feats)
        return  x if extracted_feat_indices==[] else extracted_feats