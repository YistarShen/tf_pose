
from collections.abc import Callable 
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import tensorflow as tf
from lib.models.backbones.base_backbone import BaseBackbone
from lib.Registers import MODELS
from lib.models.layers.base_cnn import Conv2D_BN_SiLU
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, concatenate, Input, Activation
import typing_extensions as tx



#---------------------------------------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------------------------------------
class MP_Conv2DBNSiLU(tf.keras.layers.Layer):
    def __init__(self, transition_channels, use_bias=False, name=None, **kwargs):
        super(MP_Conv2DBNSiLU, self).__init__(name=name)
        self.filters = ( transition_channels//2 )

    def build(self, input_shape):
        self.B1_mp = MaxPooling2D((2, 2), strides=(2, 2))
        self.B1_conv_pw = Conv2D_BN_SiLU(self.filters, kernel_size=(1,1), strides=(1, 1), use_bias=False, name=None)
        self.B2_conv_pw = Conv2D_BN_SiLU(self.filters, kernel_size=(1,1), strides=(1, 1), use_bias=False, name=None)
        self.B2_conv = Conv2D_BN_SiLU(self.filters, kernel_size=(3,3), strides=(2, 2), use_bias=False, name=None)


    def call(self, x):
        x1 = self.B1_mp(x) 
        x1 = self.B1_conv_pw(x1) 
        x2 = self.B2_conv_pw(x) 
        x2 = self.B2_conv(x2) 
        out = concatenate([x1, x2],axis=-1)
        return out

    def get_config(self):
        config = super(MP_Conv2DBNSiLU, self).get_config()
        config.update(
                {
                "filters": self.filters,
                }
        )
        return config  


#---------------------------------------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------------------------------------
class ELAN(tf.keras.layers.Layer):
    def __init__(self, in_planes, filters, name=None, **kwargs):
        super(ELAN, self).__init__(name=name)
        self.in_planes = in_planes
        self.filters = filters

    def build(self, input_shape):
        #n, h, w, c = input_shape.as_list()
        #filter = c//2
        self.B1_conv_pw = Conv2D_BN_SiLU(self.in_planes , kernel_size=(1,1), strides=(1, 1), use_bias=False, name=None)
        self.B2_conv_pw = Conv2D_BN_SiLU(self.in_planes , kernel_size=(1,1), strides=(1, 1), use_bias=False, name=None)
        self.B2_conv_1 = Conv2D_BN_SiLU(self.in_planes , kernel_size=(3,3), strides=(1, 1), use_bias=False, name=None)
        self.B2_conv_2 = Conv2D_BN_SiLU(self.in_planes , kernel_size=(3,3), strides=(1, 1), use_bias=False, name=None)
        self.B2_conv_3 = Conv2D_BN_SiLU(self.in_planes , kernel_size=(3,3), strides=(1, 1), use_bias=False, name=None)
        self.B2_conv_4 = Conv2D_BN_SiLU(self.in_planes , kernel_size=(3,3), strides=(1, 1), use_bias=False, name=None)
        self.conv_pw =  Conv2D_BN_SiLU(self.filters, kernel_size=(1,1), strides=(1, 1), use_bias=False, name=None)
    
    def call(self, x):
        feats = []
        x1 = self.B1_conv_pw(x) 
        feats.append(x1)
        x2 = self.B2_conv_pw(x)
        feats.append(x2)
        x2 = self.B2_conv_1(x2)
        x2 = self.B2_conv_2(x2)
        feats.append(x2)
        x2 = self.B2_conv_3(x2)
        x2 = self.B2_conv_4(x2)
        feats.append(x2)

        out = concatenate(feats,axis=-1)

        out = self.conv_pw(out)
        return out
    
    def get_config(self):
        config = super(ELAN, self).get_config()
        config.update(
                {
                "in_planes" : self.in_planes,
                "filters": self.filters,
                }
        )
        return config  

#---------------------------------------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------------------------------------
class ELAN_Lite(tf.keras.layers.Layer):
    def __init__(self, in_planes, filters, name=None, **kwargs):
        super(ELAN_Lite, self).__init__(name=name)
        self.in_planes = in_planes
        self.filters = filters

    def build(self, input_shape):
        #n, h, w, c = input_shape.as_list()
        #filter = c//2
        self.B1_conv_pw = Conv2D_BN_SiLU(self.in_planes , kernel_size=(1,1), strides=(1, 1), use_bias=False, name=None)
        self.B2_conv_pw = Conv2D_BN_SiLU(self.in_planes , kernel_size=(1,1), strides=(1, 1), use_bias=False, name=None)
        self.B2_conv_1 = Conv2D_BN_SiLU(self.in_planes , kernel_size=(3,3), strides=(1, 1), use_bias=False, name=None)
        self.B2_conv_2 = Conv2D_BN_SiLU(self.in_planes , kernel_size=(3,3), strides=(1, 1), use_bias=False, name=None)
        self.conv_pw =  Conv2D_BN_SiLU(self.filters, kernel_size=(1,1), strides=(1, 1), use_bias=False, name=None)
    
    def call(self, x):
        feats = []
        x1 = self.B1_conv_pw(x) 
        feats.append(x1)
        x2 = self.B2_conv_pw(x)
        feats.append(x2)
        x2 = self.B2_conv_1(x2)
        feats.append(x2)
        x2 = self.B2_conv_2(x2)
        feats.append(x2)

        out = concatenate(feats,axis=-1)

        out = self.conv_pw(out)
        return out
    
    def get_config(self):
        config = super(ELAN_Lite, self).get_config()
        config.update(
                {
                "in_planes" : self.in_planes,
                "filters": self.filters,
                }
        )
        return config
    


#---------------------------------------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------------------------------------
@MODELS.register_module()
class DarkNet(BaseBackbone):
    def __init__(self,      
            model_input_shape : Tuple[int,int]=(256,192),
            arch :  str = "YoloV7_Tiny", 
            data_preprocessor: dict = None,
            *args, **kwargs):
        #assert arch
        self.arch = arch

        super().__init__(input_size =  (*model_input_shape,3),
                        data_preprocessor = data_preprocessor,
                        name = 'DarkNet',**kwargs)

    def forward(self, x:tf.Tensor)->tf.Tensor:

        if self.arch=='YoloV7_Tiny':
            '''
            stem "  P1 and P2
            [-1, 1, Conv, [32, 3, 2]],  # 0-P1/2  
            [-1, 1, Conv, [64, 3, 2]],  # 1-P2/4    
            '''
            x = Conv2D_BN_SiLU(filters=32, kernel_size=(3,3), strides=(2,2), use_bias=False,name='P1/CBS')(x)     #(b, 320, 320, 32)
            x = Conv2D_BN_SiLU(filters=32*2, kernel_size=(3,3), strides=(2,2), use_bias=False, name='P2/CBS')(x)   #(b, 160, 160, 64)

            '''
            [-1, 1, Conv, [32, 1, 1]],
            [-2, 1, Conv, [32, 1, 1]],
            [-1, 1, Conv, [32, 3, 1]],
            [-1, 1, Conv, [32, 3, 1]],
            [[-1, -2, -3, -4], 1, Concat, [1]],
            [-1, 1, Conv, [64, 1, 1]],  # 7
            '''
            x = ELAN_Lite(in_planes=x.shape[-1]//2, filters=x.shape[-1], name='P2/ELAN_Lite')(x) #(b,160,160, 64)
            '''
            [-1, 1, MP, []],  # 8-P3/8
            [-1, 1, Conv, [64, 1, 1]],
            [-2, 1, Conv, [64, 1, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [[-1, -2, -3, -4], 1, Concat, [1]],
            [-1, 1, Conv, [128, 1, 1]],  # 14
            '''
            x = MP_Conv2DBNSiLU(transition_channels=x.shape[-1],name='P3/MP')(x)                #(b,80,80,64)
            x = ELAN_Lite(in_planes=x.shape[-1], filters=x.shape[-1]*2, name='P3/ELAN_Lite')(x) #(b,80,80,64)=>(b,80,80,128) #14

            x = tf.keras.layers.Layer(name=f'feat_P3-{x.shape[1]}x{x.shape[2]}')(x)             #feat_P3-80x80
            #feat1 = x #P3
            '''
            [-1, 1, MP, []],  # 15-P4/16
            [-1, 1, Conv, [128, 1, 1]],
            [-2, 1, Conv, [128, 1, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [[-1, -2, -3, -4], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],  # 21
            '''
            x = MP_Conv2DBNSiLU(transition_channels=x.shape[-1],name='P4/MP')(x)                 #(b,40,40,128)
            x = ELAN_Lite(in_planes=x.shape[-1], filters=x.shape[-1]*2, name='P4/ELAN_Lite')(x)  #(b,40,40,256) #21
            x = tf.keras.layers.Layer(name=f'feat_P4-{x.shape[1]}x{x.shape[2]}')(x)              #feat_P4-40x40
            #feat2 = x #P4
            '''
            [-1, 1, MP, []],  # 22-P5/32
            [-1, 1, Conv, [256, 1, 1]],
            [-2, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [[-1, -2, -3, -4], 1, Concat, [1]],
            [-1, 1, Conv, [512, 1, 1]],  # 28
            '''
            x = MP_Conv2DBNSiLU(transition_channels=x.shape[-1],name='P5/MP')(x)                #(b,20,20, 256)
            x = ELAN_Lite(in_planes=x.shape[-1], filters=x.shape[-1]*2, name='P5/ELAN_Lite')(x) #(b,20,20, 512)
            out = tf.keras.layers.Layer(name=f'feat_P5-{x.shape[1]}x{x.shape[2]}')(x)           #feat_P5-20x20

        elif self.arch =='YoloV7_Base':
            '''
            [-1, 1, Conv, [32, 3, 1]],   # 0
            [-1, 1, Conv, [64, 3, 2]],   # 1-P1/2      
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [128, 3, 2]],  # 3-P2/4  
            '''
            x = Conv2D_BN_SiLU(filters=32, kernel_size=(3,3), strides=(1,1), use_bias=False,name='P0/CBS')(x)    #(b, 640, 640, 32)
            x = Conv2D_BN_SiLU(filters=64, kernel_size=(3,3), strides=(2,2), use_bias=False,name='P1/CBS_1')(x)  #(b, 320, 320, 64)
            x = Conv2D_BN_SiLU(filters=64, kernel_size=(3,3), strides=(1,1), use_bias=False,name='P1/CBS_2')(x)  #(b, 320, 320, 64)
            x = Conv2D_BN_SiLU(filters=128, kernel_size=(3,3), strides=(2,2), use_bias=False, name='P2/CBS')(x)  #(b, 160, 160, 128)
            '''
            [-1, 1, Conv, [128, 3, 2]],  # 3-P2/4  
            [-1, 1, Conv, [64, 1, 1]],
            [-2, 1, Conv, [64, 1, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [[-1, -3, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],  # 11
            '''
            x = ELAN(in_planes=x.shape[-1]//2, filters=x.shape[-1], name='P2/ELAN')(x) #(b,160,160, 256)
            '''
            feat_P3

            ======MP_Conv2DBNSiLU========
            [-1, 1, MP, []],
            [-1, 1, Conv, [128, 1, 1]],
            [-3, 1, Conv, [128, 1, 1]],
            [-1, 1, Conv, [128, 3, 2]],
            [[-1, -3], 1, Concat, [1]],  # 16-P3/8 

            ======ELAN========
            [-1, 1, Conv, [128, 1, 1]],
            [-2, 1, Conv, [128, 1, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [[-1, -3, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [512, 1, 1]],  # 24  

            '''
            x = MP_Conv2DBNSiLU(transition_channels=x.shape[-1]//2,name='P3/MP')(x)   #(b,80,80,256)
            x = ELAN(in_planes=x.shape[-1], filters=x.shape[-1]*2, name='P3/ELAN')(x) #(b,80,80,128)=>(b,80,80,512) #14
            x = tf.keras.layers.Layer(name=f'feat_P3-{x.shape[1]}x{x.shape[2]}')(x)   #(b,80,80,512)

            '''
            feat_P4

            ======MP_Conv2DBNSiLU========
            [-1, 1, MP, []],
            [-1, 1, Conv, [256, 1, 1]],
            [-3, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [256, 3, 2]],
            [[-1, -3], 1, Concat, [1]],  # 29-P4/16  

             ======ELAN========
            [-1, 1, Conv, [256, 1, 1]],
            [-2, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [[-1, -3, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [1024, 1, 1]],     # 37            
            
            '''
            x = MP_Conv2DBNSiLU(transition_channels=x.shape[-1],name='P4/MP')(x)       #(b,80,80,512)
            x = ELAN(in_planes=x.shape[-1], filters=x.shape[-1]*2, name='P4/ELAN')(x)  #(b,40,40,1024) #21
            x = tf.keras.layers.Layer(name=f'feat_P4-{x.shape[1]}x{x.shape[2]}')(x)    #(b,40,40,1024) #21

            '''
            feat_P5

            [-1, 1, MP, []],
            [-1, 1, Conv, [512, 1, 1]],
            [-3, 1, Conv, [512, 1, 1]],
            [-1, 1, Conv, [512, 3, 2]],
            [[-1, -3], 1, Concat, [1]],  # 42-P5/32  
            [-1, 1, Conv, [256, 1, 1]],
            [-2, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [[-1, -3, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [1024, 1, 1]],  # 50          
            '''

            x = MP_Conv2DBNSiLU(transition_channels=x.shape[-1],name='P5/MP')(x)         #(b,20,20, 1024)
            x = ELAN(in_planes=x.shape[-1], filters=x.shape[-1], name='P5/ELAN')(x)      #(b,20,20, 1024)
            out = tf.keras.layers.Layer(name=f'feat_P5-{x.shape[1]}x{x.shape[2]}')(x)     #(b,20,20, 1024)

        else:
            raise RuntimeError(f"no support config of DarkNet : {self.arch}, \
                                only support 'YoloV7_Tiny' and 'YoloV7_Base' ") 

        return out

        
  



        