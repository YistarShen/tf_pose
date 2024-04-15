from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from lib.Registers import MODELS
from tensorflow.keras.layers import Concatenate, UpSampling2D
from tensorflow import Tensor
from lib.models.modules import BaseModule,ELAN
from lib.layers import Conv2D_BN
from lib.layers.convolutional import MaxPoolAndStrideConv2D

#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
@MODELS.register_module()
class PathAggregationFPN_ELAN(BaseModule):
    r"""PAFPN_ELAN ( Path Aggregation FPN with ELAN blocks)


    # yolov7                                                       
    # 51: p5 512 ---+---------------------+-> 101: out2 512        
    #               v [up 256 -> concat]  ^ [down 512 -> concat]    
    # 37: p4 1024 -> 63: p4p5 256 -------> 88: out1 256             
    #               v [up 128 -> concat]  ^ [down 256 -> concat]    
    # 24: p3 512 --> 75: p3p4p5 128 ------+--> 75: out0 128        
    #                                                               
                                                            
    Args:
            fpn_channel_ratio (float) : ratio of input features to fpn, default to 0.25
            stack_in_planes (list) : out_channels  of 1st and 2nd convs in ELAN blocks , default to [64, 128, 256, 256] 
            fpn_stack_concats (list) : extract feature's id in each ELAN block , default to [-1, -3, -5, -6],
            fpn_stack_depth (int) : depth of  ELAN block (how many conv used in one block) , default to 6
            fpn_stack_mid_ratio (float) : ratio of in_planes to hidden channels(filters of 3~6ith convs), defaults to 1. 
            activation (str) : activation used in ConvModule, defaults to 'silu'.
            only_topdown (bool) : whether use bottom up network, if Ture, fpn only use topdown network. default to False.
            bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
            bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
            activation (str) :  activation used in conv_bn blocks, defaults to 'relu'.
            data_preprocessor: dict = None,
            
    
    References:
            - [YoloV7 paper : Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors] (https://arxiv.org/abs/2207.02696)
            - [config of YoloV7_Base @WongKinYiu] (https://github.com/WongKinYiu/yolov7/blob/main/cfg/training/yolov7.yaml)
            - [config of YoloV7_Tiny @WongKinYiu] (https://github.com/WongKinYiu/yolov7/blob/main/cfg/training/yolov7-tiny.yaml)
            - [Based on implementation of YOLOV7Backbone @leondgarse's keras_cv_attention_models  ] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/yolov7/yolov7.py)
        

    Example: 
        YoloV7_Tiny - {
                        fpn_channel_ratio = 0.5,
                        fpn_stack_depth = 4 ,  
                        fpn_stack_concats =[-1, -2, -3, -4],
                        fpn_stack_in_ratio = 0.25,
                        fpn_stack_mid_ratio = 1.0,
                        activation = 'silu',
                        simple_downsample = False
        }
        feats_in : [ (80,80,128) , (40,40,256) , (20,20, 512//2) ]
        fpn_out : [(80,80,64) , (40,40,128) , (20,20,256)]

                  
        YoloV7 -  { 

                        fpn_channel_ratio = 0.25,
                        fpn_stack_depth = 6 ,  
                        fpn_stack_concats =[-1, -2, -3, -4, -5, -6],
                        fpn_stack_in_ratio = 0.5,
                        fpn_stack_mid_ratio = 0.5,
                        activation = 'swish',
                        simple_downsample = False
        }
        feats_in : [ (80,80,512) , (40,40,1024) , (20,20, 1024//2) ]
        fpn_out :  [ (80,80,128) , (40,40,256) , (20,20, 512) ]

        YoloV7_X -  { 

                        fpn_channel_ratio = 0.25,
                        fpn_stack_depth = 8 ,  
                        fpn_stack_concats = [-1, -3, -5, -7, -8],
                        fpn_stack_in_ratio = 0.5,
                        fpn_stack_mid_ratio = 1.0,
                        activation = 'swish',
                        simple_downsample = False
        }

        feats_in : [ P3:(80,80,640), P4:(40,40,1280) , P5:(20,20, 1280//2 )]
        fpn_out :  [ P3:(80,80,160), P4:(40,40,320) , P5:(20,20, 640 )]

 
    """
    def __init__(self,
            fpn_channel_ratio : int = 0.25,
            fpn_stack_concats : Optional[List[int]] = None,
            fpn_stack_depth : int = 6,
            fpn_stack_in_ratio : float = 0.5,
            fpn_stack_mid_ratio : float = 0.5,
            activation : str = 'silu',
            only_topdown : bool = False,
            simple_downsample : bool = False,
            name=None, 
            **kwargs):
            super().__init__(name=name, 
                            activation=activation, 
                            **kwargs)

            self.fpn_channel_ratio = fpn_channel_ratio
            self.fpn_stack_concats = fpn_stack_concats
            self.fpn_stack_depth = fpn_stack_depth
            self.fpn_stack_mid_ratio = fpn_stack_mid_ratio
            self.fpn_stack_in_ratio = fpn_stack_in_ratio
        
            'basic config'
            self.half_mode = only_topdown
            self.simple_downsample = simple_downsample


    def  call(self, features):
        
        if not isinstance(features, (tuple,list)):
            raise TypeError('type of features must be tuple or list')
        
        if len(features)<2:
            raise ValueError('features must have two leat features')   
        '''
        features(P3P4P5) : [ P3:(80,80,512) , P4:(40,40,1024) , P5:(20,20, 1024//2) ]
        '''
        upsamples = [features[-1]]
        state =self.name+f"PAFPN_Up_p{len(features) + 2}"
        for idx, feat in enumerate(features[:-1][::-1]):

            feat_to_up = f'p{len(features)+2-idx}'
            feat_to_fuse = f'p{len(features)+1-idx}'# feat
            state += f"p{len(features)+1-idx}"

            '''
            Layer Name : PAFPN_Up_p5p4_p5_conv (Conv2D_BN)   
                         means conv(p5) in the part of p5 upsample to p4 in PAFPN Upsample bracnch(topdown) 
            '''

            feat = Conv2D_BN(filters = feat.shape[-1]*self.fpn_channel_ratio,
                            kernel_size=1,
                            strides=1,
                            bn_epsilon = self.bn_epsilon,
                            bn_momentum = self.bn_momentum,
                            activation = self.act_name,
                            name=f'{state}_{feat_to_fuse}_conv')(feat)
            
            nn = Conv2D_BN(filters = feat.shape[-1],
                            kernel_size=1,
                            strides=1,
                            bn_epsilon = self.bn_epsilon,
                            bn_momentum = self.bn_momentum,
                            activation = self.act_name,
                            name=f'{state}_{feat_to_up}_conv')(upsamples[-1])
            
            nn_upsample = UpSampling2D(interpolation='nearest', 
                                       name=f'{state}_{feat_to_up}_UpSampling2D')(nn) #(b,20,20,128) => (b,40,40,128)
            
            feat = Concatenate(axis=-1, name=f'{state}_concat')([feat, nn_upsample]) #(b,40,40,256)

            feat = ELAN(in_planes = feat.shape[-1]*self.fpn_stack_in_ratio, 
                        out_channels = feat.shape[-1]//2,  
                        mid_ratio = self.fpn_stack_mid_ratio,
                        stack_depth =  self.fpn_stack_depth,
                        stack_concats = None,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        activation = self.act_name,
                        name=f'{state}_ELAN')(feat)

            upsamples.append(feat)

        if self.half_mode :
            #upsamples(P5P4P3) : [ P5:(20,20, 512),  P4:(40,40,256), P3:(80,80,128) ] => [P3,p4,P5]
            return upsamples[::-1]
        
        #upsamples(P5P4P3) : [ P5:(20,20, 512),  P4:(40,40,256), P3:(80,80,128) ]
        #upsamples[:-1][::-1] : [ P4:(40,40,256), P5:(20,20, 512)]  and downsamples :  [P3:(80,80,128)]
        downsamples = [upsamples[-1]]
        state =self.name + f"PAFPN_Down_p{len(features)}"
        for idx, feat in enumerate(upsamples[:-1][::-1]): 

            feat_to_down = f'p{len(features)+idx}'
            feat_to_fuse = f'p{len(features)+idx+1}' # feat
            state += f"p{len(features)+1+idx}"

            if self.simple_downsample:
                nn_downsample = Conv2D_BN(filters = downsamples[-1].shape[-1]*2,
                                kernel_size=3,
                                strides=2,
                                bn_epsilon = self.bn_epsilon,
                                bn_momentum = self.bn_momentum,
                                activation = self.act_name,
                                name=f'{state}_{feat_to_down}_conv')(downsamples[-1])              

            else:
                nn_downsample = MaxPoolAndStrideConv2D(in_channels_ratio = 1.0,
                                            strides=2, 
                                            use_bias =False,
                                            bn_epsilon = self.bn_epsilon,
                                            bn_momentum = self.bn_momentum,
                                            activation = self.act_name,
                                            name=f'{state}_{feat_to_down}_MP')(downsamples[-1])   
            

            feat = Concatenate(axis=-1, name=f'{state}_concat')([feat, nn_downsample])

            feat = ELAN(in_planes = feat.shape[-1]*self.fpn_stack_in_ratio, 
                        out_channels = feat.shape[-1]//2,  
                        mid_ratio = self.fpn_stack_mid_ratio,
                        stack_depth =  self.fpn_stack_depth,
                        stack_concats = None,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        activation = self.act_name,
                        name=f'{state}_ELAN')(feat)
            
            downsamples.append(feat)

        #downsamples(P3P4P5) : [  P3:(80,80,128), P4:(40,40,256), P5:(20,20, 256) ]

        return downsamples