from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from lib.Registers import MODELS
from tensorflow import Tensor
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, UpSampling2D
from lib.layers import SPPELAN, ADown
from lib.models.modules import BaseModule,RepNCSPELAN4

#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
@MODELS.register_module()
class PathAggregationFPN_RepNCSPELAN4(BaseModule):
    VERSION = '1.0.0'
    r"""Implements path aggregation fpn (pafpn) for object detection used in YOLOv8



                csp_depth       P3P4P5_in_channels      P3P4P5_out_channels   P3P4P5_shape
    ------------------------------------------------------------------------------------------
    YOLOv8-x       3            [320, 640, 640]         [320, 640, 640]       [80, 40, 20]
    YOLOv8-l       3            [256, 512, 512]         [256, 512, 512]       [80, 40, 20]
    YOLOv8-m       2            [192, 384, 576]         [192, 384, 576]       [80, 40, 20]
    YOLOv8-s       1            [128, 256, 512]         [128, 256, 512]       [80, 40, 20]
    YOLOv8-n       1            [ 64, 128, 256]         [ 64, 128, 256]       [80, 40, 20]



    """
    def __init__(self,
        csp_depth : int = 1,
        only_topdown : bool = False,
        use_spp_elan : bool = True,
        activation : str='silu',
        name=None, 
        **kwargs
    ):
        super().__init__(
            name=name, activation=activation,  **kwargs
        )
        self.csp_depth = csp_depth
        'basic config'
        self.use_spp_elan = use_spp_elan
        self.half_mode = only_topdown
       
    def call(self, features):
        """ 
        in_features : [p3 (80,80,512), p4 (40,40,512), p5 (20,20,512)]
        out_features : [p3 (80,80,256), p4 (40,40,512), p5 (20,20,512)]
        """
        feat = features[-1]
        state =self.name+f"PAFPN_Up_p{len(features) + 2}"

        if self.use_spp_elan :
            feat = SPPELAN(
                out_channels = feat.shape[-1] ,
                pool_size  = (5, 5),
                strides = 1,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                activation = self.act_name,
                name = f'{state}p4_p5_SPPELAN'
            )(feat)
        upsamples = [feat]



        for idx, feat in enumerate(features[:-1][::-1]):
            feat_to_up = f'p{len(features)+2-idx}'
            state += f"p{len(features)+1-idx}"

            nn = UpSampling2D(
                interpolation='nearest', 
                name=f'{state}_{feat_to_up}_UpSampling2D'
            )(upsamples[-1]) #(b,20,20,128) => (b,40,40,128)
            
            nn = Concatenate(
                axis=-1, name=f'{state}_concat'
            )([feat, nn]) #(b,40,40,256)

            filters = feat.shape[-1]//2 if idx==len(features[:-1])-1 else feat.shape[-1]
            feat = RepNCSPELAN4(
                    out_channels = filters,
                    hidden_channels = filters,
                    csp_depthes  = self.csp_depth,
                    csp_exapnd_ratio  = 0.5,
                    kernel_sizes = [3,3],
                    use_shortcut =True,
                    use_depthwise = False,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy = self.deploy,
                    name  =f'{state}_RepNCSPELAN4',  
            )(nn) 
            upsamples.append(feat)
        ##upsamples(P5P4P3) : [ P5:(20,20, 512),  P4:(40,40,256),  P3:(80,80,128) ]
        if self.half_mode :
            return upsamples[::-1]
        
        #-------------------------bottomup block------------------------------#
        downsamples = [upsamples[-1]]
        state =self.name + f"PAFPN_Down_p{len(features)}"
        for idx, feat in enumerate(upsamples[:-1][::-1]): 
            feat_to_down = f'p{len(features)+idx}'
            state += f"p{len(features)+1+idx}"

            nn = ADown(
                    out_channels =  downsamples[-1].shape[-1],
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    name=f'{state}_{feat_to_down}_MP'
            )(downsamples[-1]) 

            nn = Concatenate(
                axis=-1, name=f'{state}_concat'
            )([feat, nn]) 

            feat = RepNCSPELAN4(
                    out_channels = feat.shape[-1],
                    hidden_channels = feat.shape[-1],
                    csp_depthes  = self.csp_depth,
                    csp_exapnd_ratio  = 0.5,
                    kernel_sizes = [3,3],
                    use_shortcut =True,
                    use_depthwise = False,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy = self.deploy,
                    name  =f'{state}_RepNCSPELAN4',  
            )(nn) 
            downsamples.append(feat)
            
        #downsamples(P3P4P5) : [  P3:(80,80,128), P4:(40,40,256), P5:(20,20, 256) ]
        return downsamples