from lib.models.backbones.base_backbone import BaseBackbone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from lib.layers import Conv2D_BN
from lib.layers.convolutional import SPPCSPC, MaxPoolAndStrideConv2D
from lib.models.modules import ELAN
from lib.Registers import MODELS
import tensorflow as tf
@MODELS.register_module()
class  YOLOv7Backbone(BaseBackbone):
    r""" YOLOV7Backbone

    Based on paper's introduction, YOLOV7Backbone doesn't need pretrained weights  

    Args:
            model_input_shape (Tuple[int,int]) : default to (640,640)
            stem_type (str) : determine which stem's type use to downsample to P1 (1/2) from input shape 
            stem_width (int) : stem's channels (P1's channels) default to 64.
            channels ( List[int] ) : not use
            stack_in_planes (list) : used filters in 1st and 2nd convs of ELAN blocks , default to [64, 128, 256, 256] 
            stack_concats (list) : extract feature's id in each ELAN block , default to [-1, -3, -5, -6],
            stack_depth (int) : depth of  ELAN block (how many conv used in one block) , default to 6
            stack_out_ratio (float) : expand ratio of output conv in ELAN block , default to 1.
                                      output_channels in ith ELAN block  = len(stack_concats)*stack_in_planes[i]*stack_out_ratio 
            SPPCSPC_depth (int) : spp depth, default to 2. 
                                  if SPPCSPC_depth=0, SPPCSPC block is invalid 
            out_feat_indices (List[int]) : Output from which stages,  default to None 
                                           out_feat_indices=[3,4,5] that means model outputs [P3,P4,P5], here P2 means 1/4 scale of input image.
                                           Support negative indices, i.e. [-3,-2,-1] means [P3,P4,P5]
                                           it can be noted that the order of feats must be from larger scale to small , i.e. P3->P5 , and P5->P3 is invalid
            bn_epsilon (float) : default to BATCH_NORM_EPSILON .
            bn_momentum (float) : default to BATCH_NORM_MOMENTUM.
            activation ('swish') :  default to 'swish'
            data_preprocessor: dict = None,
            
    
    References:
            - [YoloV7 paper : Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors] (https://arxiv.org/abs/2207.02696)
            - [config of YoloV7_Base @WongKinYiu] (https://github.com/WongKinYiu/yolov7/blob/main/cfg/training/yolov7.yaml)
            - [config of YoloV7_Tiny @WongKinYiu] (https://github.com/WongKinYiu/yolov7/blob/main/cfg/training/yolov7-tiny.yaml)
            - [Based on implementation of YOLOV7Backbone @leondgarse's keras_cv_attention_models  ] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/yolov7/yolov7.py)

    Example:
        '''Python

        model = YOLOv7Backbone(model_input_shape=(640,640),
                            stem_width = 32,
                            stem_type = 'Tiny',
                            channels  = [32, 64, 128, 256, 512],
                            stack_in_planes = [32, 64, 128, 256],
                            stack_concats = [-1, -2,-3,-4],
                            stack_depth = 4,
                            stack_out_ratio  = 0.5,
                            SPPCSPC_depth = 1,
                            activation = 'silu',
                            data_preprocessor= dict(type='ImgNormalization', 
                                                    img_mean = [0.485, 0.456, 0.406],  
                                                    img_std = [0.229, 0.224, 0.225]
                                                    )
                            )

        
        YoloV7_Tiny -  { 
                        stem_type = 'Tiny',
                        stem_width = 32,
                        channels =     [64,128,256,512]
                        stack_in_planes  =  [32, 64, 128, 256]
                        stack_depth=4 ,  
                        stack_concats= [-1, -2,-,3,-4],
                        stack_out_ratio = 0.5,
                        out_feat_indices = [3,4,5],
                        activation = 'silu'
                        SPPCSPC_depth = 1
                    }
                channels per stage :
                [ P1:(320,320,32), P2:(160,160,64), P3:(80,80,128), P4:(40,40,256) , P5:(20,20, 512//2 ) with SPP ]
                [ P1:(320,320,32), P2:(160,160,64), P3:(80,80,128), P4:(40,40,256) , P5:(20,20, 512) w/o SPP]  
                   
        YoloV7 -  { 
                        stem_type = 'Base',
                        stem_width = 64,
                        channels = [256,512,1024,1024],
                        stack_in_planes  = [64, 128, 256, 256],
                        stack_depth=6 ,  
                        stack_concats =[-1, -3, -5, -6],
                        stack_out_ratio = 1.0,
                        out_feat_indices = [3,4,5],
                        activation = 'swish',
                        SPPCSPC_depth = 2
                   }
                channels per stage :
                [ P1:(320,320,64), P2:(160,160,256), P3:(80,80,512), P4:(40,40,1024) , P5:(20,20, 1024//2 ) with SPP ]
                [ P1:(320,320,64), P2:(160,160,256), P3:(80,80,512), P4:(40,40,1024) , P5:(20,20, 1024) w/o SPP]  

        YoloV7_X -  { 
                        stem_type = 'Base'
                        stem_width = 80,
                        channels = [256,512,1024,1024],
                        stack_in_planes  = [64, 128, 256, 256],
                        stack_depth=8 ,  
                        stack_concats =[-1, -3, -5, -7, -8]
                        stack_out_ratio = 1.0,
                        out_feat_indices = [3,4,5],
                        activation = 'swish',
                        SPPCSPC_depth = 2
                   }
                channels per stage :
                [ P1:(320,320,80), P2:(160,160,320), P3:(80,80,640), P4:(40,40,1280) , P5:(20,20, 1280//2 ) with SPP ]
                [ P1:(320,320,80), P2:(160,160,320), P3:(80,80,640), P4:(40,40,1280) , P5:(20,20, 1280) w/o SPP]         

        YoloV7 -  { stack_depth=6 ,  mid_ratio=1.0, stack_concats= [-1, -3, -5, -6] } in backbone,
                  { stack_depth=6 ,  mid_ratio=0.5, stack_concats= [-1,-2,-3,-4, -5, -6] }  in pafpn (W_ELAN)

    """
    def __init__(self,      
            model_input_shape : Tuple[int,int]=(256,192),
            stem_type : str = 'base',
            stem_width : int = 64,
            channels : List[int] = [256,512,1024,1024],
            stack_in_planes : list = [64, 128, 256, 256],
            stack_concats : list = [-1, -3, -5, -6],
            stack_depth : int= 6,
            stack_out_ratio : float = 1.0,
            SPPCSPC_depth : int = 2,
            out_feat_indices : Optional[List[int]] = None,
            data_preprocessor: dict = None,
            *args, **kwargs):
        
        self.channels = channels

        self.stem_type = stem_type
        self.stem_width = stem_width

        'elan config'
        self.stack_in_planes = stack_in_planes
        self.stack_concats = stack_concats
        self.stack_depth = stack_depth
        self.stack_out_ratio = stack_out_ratio
        'SPPCSPC'
        self.SPPCSPC_depth = SPPCSPC_depth
        
        'basic config'
        self.out_feat_indices = out_feat_indices
        #self.bn_epsilon = bn_epsilon
        #self.bn_momentum = bn_momentum
        #self.act = activation  
    
        super().__init__(input_size =  (*model_input_shape,3),
                        data_preprocessor = data_preprocessor,
                        name = 'YOLOv7Backbone',**kwargs)
        
         
    def call(self,  x:tf.Tensor)->tf.Tensor:
        
        extracted_feat_indices = [] if self.out_feat_indices is None else self.out_feat_indices
        if extracted_feat_indices != [] : extracted_feats = [] 
          
        #self.feats = []
        if self.stem_type =='Tiny':
            x = Conv2D_BN(filters = self.stem_width,
                        kernel_size=3,
                        strides=2,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        activation = self.act_name,
                        name='Stem_P1_CBS')(x)
            
        else:
            x = Conv2D_BN(filters = self.stem_width//2,
                            kernel_size=3,
                            strides=1,
                            bn_epsilon = self.bn_epsilon,
                            bn_momentum = self.bn_momentum,
                            activation = self.act_name,
                            name='Stem_P0_Conv')(x)
            
            x = Conv2D_BN(filters = self.stem_width,
                            kernel_size=3,
                            strides=2,
                            bn_epsilon = self.bn_epsilon,
                            bn_momentum = self.bn_momentum,
                            activation = self.act_name,
                            name='Stem_P1_Conv1')(x)

            x = Conv2D_BN(filters = x.shape[-1],
                            kernel_size=3,
                            strides=1,
                            bn_epsilon = self.bn_epsilon,
                            bn_momentum = self.bn_momentum,
                            activation = self.act_name,
                            name='Stem_P1_Conv2')(x)
            
            
        for idx, elan_in_planes in enumerate (self.stack_in_planes): 
                #p_feat = 'Stem_P2' if idx == 0 else f'Stage_P{idx+2}'
                p_feat = f'Stage_P{idx+2}'

                 #P3,P4,P5      
                if idx == 0 :
                    x = Conv2D_BN(filters = x.shape[-1]*2,
                                kernel_size=3,
                                strides=2,
                                bn_epsilon = self.bn_epsilon,
                                bn_momentum = self.bn_momentum,
                                activation = self.act_name,
                                name=f'{p_feat}_Conv')(x)
                else:
                    x = MaxPoolAndStrideConv2D(strides=2, 
                        use_bias =False,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        activation = self.act_name,
                        name=f'{p_feat}_MPConv')(x)    #(b,80,80,64) -> (b,80,80,64)
                
                    
                out_channels =  int(elan_in_planes * len(self.stack_concats)*self.stack_out_ratio)
                x = ELAN(in_planes = elan_in_planes, 
                            out_channels = out_channels,  #x.shape[-1] if self.arch=='YoloV7_Tiny' else x.shape[-1]*2,
                            mid_ratio = 1.0,
                            stack_depth = self.stack_depth,
                            stack_concats = self.stack_concats,
                            bn_epsilon = self.bn_epsilon,
                            bn_momentum = self.bn_momentum,
                            activation = self.act_name,
                            name=f'{p_feat}_ELAN')(x)
                
                # if idx == 3:
                #     break
                if idx == len(self.stack_in_planes)-1 and self.SPPCSPC_depth>0:
                    'final layer spp'
                    x = SPPCSPC(out_channels = x.shape[-1]//2,
                                expansion = 0.5,
                                depth = self.SPPCSPC_depth,
                                SPPFCSPC = False,
                                lite_type = False,
                                pool_sizes =(5, 9, 13),
                                bn_epsilon = self.bn_epsilon,
                                bn_momentum = self.bn_momentum,
                                activation = self.act_name,
                                name='SPPCSPC')(x)  
                
                x = tf.keras.layers.Layer(name=f'feat_P{idx+2}_{x.shape[1]}x{x.shape[2]}')(x)   #(b,80,80,512)                  
                #self.feats.append(1) 
                if (idx+2 in extracted_feat_indices) or ((idx+2)-6 in extracted_feat_indices):
                        extracted_feats.append(x)
                  
        return  x if extracted_feat_indices==[] else extracted_feats