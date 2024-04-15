from typing import  List, Optional,Tuple
import tensorflow as tf
import tensorflow_addons as tfa
from lib.models.modules import BaseModule
from lib.Registers import MODELS
from lib.layers import Conv2D_BN,DepthwiseConv2D_BN
from .heatmap_heads import  HeatmapBaseHead, PoseRefineMachine
from tensorflow.keras.layers import  Resizing, Activation, Add, Multiply



#-----------------------------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------------------------

# @MODELS.register_module()
# class PoseRefineMachine(BaseModule):
#     VERSION = '1.0.0'
#     r"""Pose Refine Machine.


                     
#     """
#     def __init__(self,
#             kernel_size : int = 3, 
#             activation : str ="relu", 
#             name='RPM', 
#             **kwargs): 
        
#         super().__init__(
#             name=name,  activation=activation,  **kwargs
#         )
#         self.kernel_size = kernel_size
        
#     def build(self, input_shape):

#         _, _,_, self. out_channels = input_shape
#         self.conv = Conv2D_BN(
#             filters = self.out_channels ,
#             kernel_size= self.kernel_size,
#             strides=1,
#             use_bias = False,
#             use_bn = True,
#             activation = self.act_name,
#             bn_epsilon = self.bn_epsilon,
#             bn_momentum = self.bn_momentum,
#             name =  self.name + 'ConvBn',
#             dtype=tf.float32
#         ) 

#         self.middle_path  = tf.keras.Sequential(
#             layers = [
#                 tfa.layers.AdaptiveAveragePooling2D(
#                     (1,1), name = 'adptAP'
#                 ),
#                 Conv2D_BN(
#                     filters = self.out_channels ,
#                     kernel_size= 1,
#                     strides=1,
#                     use_bias = False,
#                     use_bn = True,
#                     activation = self.act_name,
#                     bn_epsilon = self.bn_epsilon,
#                     bn_momentum = self.bn_momentum,
#                     name =  'ConvBn1',
#                 ) ,
#                 Conv2D_BN(
#                     filters = self.out_channels ,
#                     kernel_size= 1,
#                     strides=1,
#                     use_bias = False,
#                     use_bn = True,
#                     activation = self.act_name,
#                     bn_epsilon = self.bn_epsilon,
#                     bn_momentum = self.bn_momentum,
#                     name =  'ConvBn2',
#                 ) ,

#                 Activation('sigmoid', dtype=tf.float32)
#             ] ,
#             name = self.name+'middle_path'   
#         ) 

#         self.bottom_path   = tf.keras.Sequential(
#             layers = [
#                 Conv2D_BN(
#                     filters = self.out_channels ,
#                     kernel_size= 1,
#                     strides=1,
#                     use_bias = False,
#                     use_bn = True,
#                     activation = self.act_name,
#                     bn_epsilon = self.bn_epsilon,
#                     bn_momentum = self.bn_momentum,
#                     name =  'ConvBn',
#                 ) ,
#                 DepthwiseConv2D_BN(
#                         kernel_size=9,
#                         strides=1,
#                         use_bias = False,
#                         use_bn = True,
#                         activation = self.act_name,
#                         bn_epsilon = self.bn_epsilon,
#                         bn_momentum = self.bn_momentum,
#                         name = 'DWConvBn'
#                 ),
#                 Activation('sigmoid', dtype=tf.float32)
#             ] ,
#             name = self.name+'bottom_path'   
#         ) 

#         self.mul_1 = Multiply(
#             name= self.name + "b1_mul",dtype=tf.float32
#         )
#         self.mul_2 = Multiply(
#             name= self.name  + "b2_mul",dtype=tf.float32
#         )

#         self.add_output = Add(
#             name=self.name  +'output',  dtype=tf.float32
#         )
         
#     def call(self, x : tf.Tensor) -> tf.Tensor:

#         x = self.conv(x)  # (b,64,48,17)

#         out_1 = self.adptAP(x) # (b,1,1,17)
#         out_1 = self.middle_path(out_1) # (b,1,1,17)
#         out_2 = self.bottom_path(x) # (b,64,48,17)


#         output = self.mul_1([out_1, out_2])
#         output = self.mul_2([output, x])
#         output = self.add_output([x, output])

#         # output = Multiply(
#         #     name= self.name + "b1_mul",dtype=tf.float32
#         # )([out_1, out_2])

#         # output = Multiply(
#         #     name= self.name  + "b2_mul",dtype=tf.float32
#         # )([output, x])

#         # output = Add(
#         #     name=self.name  +'RPM_out',  dtype=tf.float32
#         # )([x, output])
        
#         return output

#-----------------------------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------------------------
@MODELS.register_module()
class AuxiliaryHeatmapHead(BaseModule):
    VERSION = '1.1.0'
    r"""AuxiliaryHeatmapHead.

    AuxiliaryHeatmapHead 
  
 
    Required Data's Values dtype:
        - tf.Tensor
    Args:
        out_shape (tuple[int]):  Shape of the output heatmaps.
        interpolation (str) : the interpolation method used by tf.keras.layers.Resizing().  
                Supports "bilinear", "nearest", "bicubic", "area", "lanczos3", 
                "lanczos5", "gaussian", "mitchellcubic". Defaults to "bilinear". Default to "bilinear"
        conv_kernel_size (int) : The kernel size of  2nd conv layer. Default to 3,
        out_channels (int):  Number of output channels of 2nd conv layer

    References:
        - [Inspired on implementation of 'PredictHeatmap' @mmpose] 
         (https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/heads/heatmap_heads/mspn_head.py)                    


    Example:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input    
        x = Input(shape=(32*24, 128))
        out = AuxiliaryHeatmapHead(out_shape  = (64,48),
                    interpolation  = 'bilinear',
                    out_channels = 17,
                    conv_kernel_size  = 3,
                    input_transforms = [  
                        dict(type="Reshape", target_shape=(32,24,-1),name="stage1_Reshape"),
                        dict(type="Conv2D", filters=128, kernel_size=1,name="stage1_pwconv")
                    ],
                    activation  ="relu",
                    bn_epsilon = 1e-5,
                    bn_momentum = 0.9,
                    name = "AuxHeatmapHead"
        )(x)
        model = Model(x,out)
        model.summary(200)
    """

    def __init__(self, 
            out_shape : Tuple[int,int] = (64,48),
            out_channels : int = 17,
            conv_kernel_size : int = 3,
            interpolation : str = 'bilinear',
            activation : str ="relu", 
            name='AuxiliaryHeatmapHead', 
            **kwargs): 
        super().__init__(
            name=name,  activation=activation,  **kwargs
        )

        #self.compute_dtype = mixed_precision.global_policy().compute_dtype
        'upsample cfg'
        self.upsample_method = interpolation
        self.out_shape = out_shape
        'output conv'
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
  


    def build(self,input_shape):

        in_channels = input_shape[-1]

        self.PreConvBN = Conv2D_BN(
                filters = in_channels,
                kernel_size=1,
                strides=1,
                use_bias = False,
                use_bn = True,
                activation = self.act_name,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                dtype = 'float32',
                name = self.name + 'PreConvBN'
        )
        self.PostConvBN = Conv2D_BN(
                filters = self.out_channels,
                kernel_size=self.conv_kernel_size,
                strides=1,
                use_bias = False,
                use_bn = True,
                activation = None,
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,
                dtype = 'float32',
                name = self.name + 'PostConvBN'
        )
        self.resize = Resizing(
                self.out_shape[0],
                self.out_shape[1], 
                name=self.name + 'Resize', 
                interpolation= self.upsample_method , 
                dtype='float32'
        )

    def call(self, x : tf.Tensor) -> tf.Tensor:
        x = self.PreConvBN(x)
        x = self.PostConvBN(x)
        heatmap = self.resize(x)
        return heatmap
    



#---------------------------------------------------------------------------
#
#---------------------------------------------------------------------------
@MODELS.register_module()
class MSPNHead(BaseModule):
    VERSION = '1.0.0'
    r"""Multi-Stage Pose estimation Network (MSPN) used in RSN

    Multi-stage multi-unit heatmap head introduced in `Multi-Stage Pose
    estimation Network (MSPN)`_ by Li et al (2019), and used by `Residual Steps
    Networks (RSN)`_ by Cai et al (2020). The head consists of multiple stages
    and each stage consists of multiple units. Each unit of each stage has some
    conv layers.


    Args:
        num_stages (int):  Number of stages. Default to 1
        num_aux_heads(int) : Number of AuxiliaryHeatmapHeads in each stages. Default to 3.
        out_shape (Tuple[int,int]) : The output shape of the output heatmaps, Default to (64,48).
        out_channels (int) : Number of output channels. Default to 17 
        prim_deconv_filters_list (List[int]):  The output channel
            number of each deconv layer in Primary Heads. Default to [256,256,256],
        prim_deconv_kernels_list (List[int]): . The kernel size
            of each deconv layer in Primary Heads. Default to  [4,4,4],
        prim_conv_kernel_size (Optional[int]) :  The kernel size of each intermediate conv layer 
            in Primary branch Heads.  Default to None,
        prim_conv_out_channels (Optional[int]) :The output channel number of each intermediate conv layer 
            in Primary branch Heads. ``None`` means no intermediate conv layer 
            between deconv layers and the final conv layer. Default to None,
        aux_interpolation (str):  The method to resize features to out_shape in Auxiliary branch Heads. 
            Default to 'bilinear'.
        aux_conv_kernel_size (int) :  The kernel size of conv layer in Auxiliary branch Heads. Default to 3.
        activation (str) : activation.  Default to "relu", 


    References:
        - [Inspired on implementation of 'MSPNHead' @mmpose] 
          (https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/heads/heatmap_heads/mspn_head.py)                    
    

    Example:
    '''Pyhton (4 stages with 3 aux_head + 1 prim_haed)

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input    
    xs_1 = [Input(shape=(64, 48, 64)),Input(shape=(32, 24, 128)) , Input(shape=(16, 12, 256)), Input(shape=(8, 6, 512))]
    xs_2 = [Input(shape=(64, 48, 64)),Input(shape=(32, 24, 128)) , Input(shape=(16, 12, 256)), Input(shape=(8, 6, 512))]
    xs_3 = [Input(shape=(64, 48, 64)),Input(shape=(32, 24, 128)) , Input(shape=(16, 12, 256)), Input(shape=(8, 6, 512))]
    xs_4 = [Input(shape=(64, 48, 64)),Input(shape=(32, 24, 128)) , Input(shape=(16, 12, 256)), Input(shape=(8, 6, 512))]
    x = [xs_1, xs_2, xs_3, xs_4]


    out = MSPNHead(
                num_stages  = 4,
                num_aux_heads  = 3,
                out_channels  = 17,
                out_shape  = (64,48),
                prim_deconv_filters_list = [],
                prim_deconv_kernels_list = [],
                prim_conv_kernel_size  = None,
                prim_conv_out_channels = None,
                aux_interpolation  = 'bilinear',
                aux_conv_kernel_size  = 3,
                activation  ="relu", 
                name = "MSPNHead"
    )(x)
    model = Model(x,out)
    model.summary(200)
                   

    """

    def __init__(self, 
            num_stages : int = 1,
            prim_head_index : int = 0,
            aux_heads_indices : List[int] = [1,2,3],
            use_prm : bool = False,
            out_shape : Tuple[int,int] = (64,48),
            out_channels : int = 17,
            prim_deconv_filters_list : List[int] = [256,256,256],
            prim_deconv_kernels_list : List[int] = [4,4,4],
            prim_conv_kernel_size : Optional[int] = None,
            prim_conv_out_channels : Optional[int] = None,
            aux_interpolation : str = 'bilinear',
            aux_conv_kernel_size : int = 3,
            activation : str ="relu", 
            name='MSPNHead', 
            **kwargs): 
        super().__init__(
            name=name,  activation=activation,  **kwargs
        )
      
        self.num_stages = num_stages
        self.out_shape = out_shape
        self.out_channels = out_channels
        self.use_prm = use_prm
        'primary head cfgs'
        self.prim_conv_kernel_size = prim_conv_kernel_size
        self.prim_conv_out_channels = prim_conv_out_channels
        self.prim_deconv_filters_list = prim_deconv_filters_list
        self.prim_deconv_kernels_list = prim_deconv_kernels_list
        'Auxiliary head cfgs'
    
        self.aux_interpolation = aux_interpolation
        self.aux_conv_kernel_size = aux_conv_kernel_size
       
        self.num_aux_heads = len(aux_heads_indices)
        self.prim_head_index = prim_head_index
        self.aux_heads_indices = aux_heads_indices
    def build(self, input_shape):


        if len(input_shape)!=self.num_stages: 
            raise ValueError(
                f"num_stages of input features must be {self.num_stages}, "
                f"but got {len(len(input_shape))} @{self.__class__.__name__}"
            )

        for j in range(self.num_stages):
            feats_shape = input_shape[j]
            num_feats = len(feats_shape)

            if num_feats!= self.num_aux_heads+1:
                raise ValueError(
                    f"num features in {j}th stage didn't match setting={self.num_aux_heads+1}"
                )
            
            if num_feats==1:
                continue

            'verify whether multi-features in each stage is descending order [P2,P3,P4,P5]'
            if not all(
                    [(a[1]==b[1]*2 and  a[2]==b[2]*2) 
                    for a, b in zip(feats_shape[:-1],feats_shape[1:])]
                ):
                raise ValueError(
                    f"the shapes of feat must be in descending order, \n"    
                    f",i.e. [P2,P3,P4,P5] not [P5,P4,P3], but got {[ X for X in (feats_shape)]} "   
                )   
            


        "init list for AuxiliaryHeatmapHeads/PrimaryHeatmapHeads"      
        if self.num_aux_heads>0 :
            self.AuxiliaryHeatmapHeads = [[None]*self.num_aux_heads]*self.num_stages

        self.PrimaryHeatmapHeads = [None]*self.num_stages

        for j in range(self.num_stages):
            ith_stage = j +1

            if hasattr(self,'AuxiliaryHeatmapHeads'):
                self.AuxiliaryHeatmapHeads[j] = [
                    AuxiliaryHeatmapHead(
                        out_shape = self.out_shape,
                        out_channels = self.out_channels,
                        interpolation =  self.aux_interpolation,
                        conv_kernel_size = self.aux_conv_kernel_size,
                        activation  = self.act_name, 
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        name=self.name+f'Stage{ith_stage}_AuxHead{i+1}'
                    ) for  i in range(self.num_aux_heads)
                ]

            self.PrimaryHeatmapHeads[j] = HeatmapBaseHead(
                    out_channels = self.out_channels,
                    deconv_filters_list=self.prim_deconv_filters_list,  
                    deconv_kernels_list = self.prim_deconv_kernels_list,
                    conv_kernel_szie = self.prim_conv_kernel_size,
                    conv_out_channels = self.prim_conv_out_channels,
                    activation =self.act_name, 
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,                   
                    input_transforms = None,
                    name=self.name+f'Stage{ith_stage}_PrimHead'  
            )
           

        if self.use_prm:
            self.prm = PoseRefineMachine(            
                kernel_size = 3, 
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,   
                activation = self.act_name, 
                name=self.name+'RPM'
            )
        

    def call(self, input_feats : List[tf.Tensor]) -> tf.Tensor:


        """
        input_feats  : [[P3,P4,P5]] / [[(b,32,24,128), (b,16,12,256), (b,8,6,512)]]
        stage2  : [[P3,P4,P5], [P3,P4,P5], [P3,P4,P5]]
                          

        input_feats = [P3,P4,P5] / [(b,32,24,128), (b,32,24,128), (b,16,12,256), (b,8,6,512)]
                       primary_feature = P3 ; auxiliary_features = [P3,P4,P5]
                        or 
                      [P2,P3,P4,P5] / [(b,64,48,128), (b,32,24,256), (b,16,12,512), (b,8,6,1024)]
                       primary_feature = P2 ; auxiliary_features = [P3,P4,P5]
                      
                    
        """

        if not isinstance(input_feats, List): 
            raise TypeError(
                "type of input_feats must be List \n" 
                f'but got {type(input_feats)} @{self.__class__.__name__}'
            )
        
        stages_multi_heatmaps = []
        for i in range(self.num_stages):

 
            'unpack data'
            primary_feature = input_feats[i][self.prim_head_index]
            #auxiliary_features = input_feats[i][1:]
            auxiliary_features = [input_feats[i][idx] for idx in self.aux_heads_indices]
            

            if len(auxiliary_features)!=self.num_aux_heads:
                raise ValueError( 
                    f"num of input auxiliary_features doesn't match to {self.num_aux_heads} "
                    f"but got { len(auxiliary_features)} @{self.__class__.__name__}"
                )
            
            primary_heatmap = self.PrimaryHeatmapHeads[i](primary_feature) 

            if i==self.num_stages-1 and hasattr(self,'prm'):
                primary_heatmap = self.prm(primary_heatmap)

            if hasattr(self,'AuxiliaryHeatmapHeads'):
                auxiliary_heatmaps = []
                for j, AuxiliaryHeatmapHead in enumerate(self.AuxiliaryHeatmapHeads[i]):
                    auxiliary_heatmaps.append(
                        AuxiliaryHeatmapHead(auxiliary_features[j])
                    )
                    assert primary_heatmap.shape== auxiliary_heatmaps[j].shape, \
                    "primary_heatmap's shape must be equal to auxiliary_feature"
                
                #multi_heatmaps = tf.stack([primary_heatmap, *auxiliary_heatmaps], axis=1)
                multi_heatmaps = tf.keras.layers.Lambda(
                    lambda x: tf.stack(x, axis=1), 
                    name = self.name+f'Stage{i+1}_out',
                    dtype= tf.float32
                )([primary_heatmap, *auxiliary_heatmaps])

            else:
                multi_heatmaps = primary_heatmap

            stages_multi_heatmaps.append(multi_heatmaps)
        return stages_multi_heatmaps
    
# #---------------------------------------------------------------------------
# #
# #---------------------------------------------------------------------------
# class TopdownMultiHeatmapBaseHead(tf.Module):
#     VERSION = '1.0.0'
#     r"""TopdownMultiHeatmapBaseHead.

#     TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers(Conv2DTranspose)
#     and a simple updample layer(UpSampling2D).
  
 
#     Required Data's Values dtype:
#         - tf.Tensor
#     Args:
#         deconv_filters_list (List): Number of filters in each Conv2DTranspose if len(List) > 0
#         deconv_kernels_list (List): kernels used in each Conv2DTranspose if len(List) > 0
#         upsample_scale (int):  upsample ratio used in UpSampling2D if upsample_scale>0
#         final_conv_kernel (int): kernels used in final Conv2D
#         final_conv_out_channels (int):  Number of output channels

#     Example:

#             - classical_hm_head_cfg = dict( econv_filters_list =[256, 256], 
#                                             deconv_kernels_list = [4,4],
#                                             upsample_scale=0,
#                                             final_conv_kernel= 1,
#                                             final_conv_out_channels = 17)
                   

#     """

#     def __init__(self, 
#             num_aux_heads : int = 3,
#             out_feat_shape : Tuple[int,int] = (64,48),
#             deconv_filters_list : List[int] = [256,256,256],
#             deconv_kernels_list : List[int] = [4,4,4],
#             final_conv_out_channels : int = 17,
#             final_conv_kernel : int = 1,
#             aux_final_conv_kernel : int = 3,
#             activation : str ="relu", 
#             bn_epsilon : float= 1e-5,
#             bn_momentum : float= 0.9,
#             name=None, 
#             **kwargs): 
#         super().__init__(name=name)

#         if num_aux_heads<1:
#             raise ValueError(f"num_aux_heads nened to be >1  @{self.__call__.__name__}") 
        
#         self.num_aux_heads = num_aux_heads


#         self.AuxiliaryHeatmapHeads = [AuxiliaryHeatmapHead(out_feat_shape = out_feat_shape,
#                                                         final_conv_out_channels = final_conv_out_channels,
#                                                         final_conv_kernel = aux_final_conv_kernel,
#                                                         activation  = activation, 
#                                                         bn_epsilon = bn_epsilon,
#                                                         bn_momentum = bn_momentum,
#                                                         name=f'AuxHead_{i+1}') for  i in range(num_aux_heads)]
        
#         self.PrimaryHeatmapHead = TopdownHeatmapBaseHead(deconv_filters_list = deconv_filters_list, 
#                                                         deconv_kernels_list = deconv_kernels_list,
#                                                         final_conv_kernel = final_conv_kernel,
#                                                         final_conv_out_channels = final_conv_out_channels,
#                                                         activation =activation, 
#                                                         input_transforms = None,
#                                                         name=self.name+f'_PrimHead')

#     def __call__(self, input_feats : List[tf.Tensor]) -> tf.Tensor:
#         """
        
#         input_feats = [P3,P4,P5] / [(b,32,24,128), (b,32,24,128), (b,16,12,256), (b,8,6,512)]
#                        primary_feature = P3 ; auxiliary_features = [P3,P4,P5]
#                         or 
#                       [P2,P3,P4,P5] / [(b,64,48,128), (b,32,24,256), (b,16,12,512), (b,8,6,1024)]
#                        primary_feature = P2 ; auxiliary_features = [P3,P4,P5]
                      
                    
#         """

#         if not isinstance(input_feats, List): 
#             raise TypeError(f'type of input_feats must be List, but got {type(input_feats)} @{self.__call__.__name__}')
        
#         if len(input_feats)<self.num_aux_heads: 
#             raise ValueError(f"input features must be {self.num_aux_heads} at least"
#                              f"but got {len(input_feats)} @{self.__call__.__name__}")
        
#         primary_feature = input_feats[0]
#         auxiliary_features = input_feats[-self.num_aux_heads:]

#         if len(auxiliary_features)!=self.num_aux_heads:
#             raise ValueError( f"num of input auxiliary_features doesn't match to {self.num_aux_heads} "
#                               f"but got { len(auxiliary_features)} @{self.__class__.__name__}")

#         auxiliary_heatmaps =[]
#         for i, AuxiliaryHeatmapHead in enumerate(self.AuxiliaryHeatmapHeads):
#             auxiliary_heatmaps.append(AuxiliaryHeatmapHead(auxiliary_features[i]))

#         primary_heatmap = self.PrimaryHeatmapHead(primary_feature) 

#         #print(primary_heatmap.shape, auxiliary_heatmaps[1].shape)
#         assert primary_heatmap.shape== auxiliary_heatmaps[0].shape, \
#         "primary_heatmap's shape must be equal to auxiliary_feature"

#         multi_heatmaps = tf.stack([primary_heatmap, *auxiliary_heatmaps], axis=1)

#         return multi_heatmaps
    
#     def get_config(self):
#         config = super().get_config()
        
#         config.update(
#             {

#                 "num_aux_heads" : self.num_aux_heads,
      
#             }
#         )
#         return config   

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)   
 

    
