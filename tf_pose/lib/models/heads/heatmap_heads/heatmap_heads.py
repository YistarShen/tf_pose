from typing import  List, Optional, Union, Tuple
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Conv2D, UpSampling2D, Activation, Multiply, Add, Reshape
from lib.models.modules import BaseModule
from lib.Registers import MODELS
from lib.layers import Conv2D_BN, DepthwiseConv2D_BN, Conv2DTranspose_BN

#-----------------------------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------------------------
@MODELS.register_module()
class PoseRefineMachine(BaseModule):
    VERSION = '1.0.0'
    r"""Pose Refine Machine.


                     
    """
    def __init__(self,
            kernel_size : int = 3, 
            activation : str ="relu", 
            name='RPM', 
            **kwargs): 
        
        super().__init__(
            name=name,  activation=activation,  **kwargs
        )
        self.kernel_size = kernel_size
        
    def build(self, input_shape):

        _, _,_, self. out_channels = input_shape
        self.conv = Conv2D_BN(
            filters = self.out_channels ,
            kernel_size= self.kernel_size,
            strides=1,
            use_bias = False,
            use_bn = True,
            activation = self.act_name,
            bn_epsilon = self.bn_epsilon,
            bn_momentum = self.bn_momentum,
            name =  self.name + 'ConvBn',
            dtype=tf.float32
        ) 

        self.middle_path  = tf.keras.Sequential(
            layers = [
                tfa.layers.AdaptiveAveragePooling2D(
                    (1,1), name = 'adptAP'
                ),
                Conv2D_BN(
                    filters = self.out_channels ,
                    kernel_size= 1,
                    strides=1,
                    use_bias = False,
                    use_bn = True,
                    activation = self.act_name,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    name =  'ConvBn1',
                ) ,
                Conv2D_BN(
                    filters = self.out_channels ,
                    kernel_size= 1,
                    strides=1,
                    use_bias = False,
                    use_bn = True,
                    activation = self.act_name,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    name =  'ConvBn2',
                ) ,

                Activation('sigmoid', dtype=tf.float32)
            ] ,
            name = self.name+'middle_path'   
        ) 

        self.bottom_path   = tf.keras.Sequential(
            layers = [
                Conv2D_BN(
                    filters = self.out_channels ,
                    kernel_size= 1,
                    strides=1,
                    use_bias = False,
                    use_bn = True,
                    activation = self.act_name,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    name =  'ConvBn',
                ) ,
                DepthwiseConv2D_BN(
                        kernel_size=9,
                        strides=1,
                        use_bias = False,
                        use_bn = True,
                        activation = self.act_name,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        name = 'DWConvBn'
                ),
                Activation('sigmoid', dtype=tf.float32)
            ] ,
            name = self.name+'bottom_path'   
        ) 

        self.mul_1 = Multiply(
            name= self.name + "b1_mul",dtype=tf.float32
        )
        self.mul_2 = Multiply(
            name= self.name  + "b2_mul",dtype=tf.float32
        )

        self.add_output = Add(
            name=self.name  +'output',  dtype=tf.float32
        )
         
    def call(self, x : tf.Tensor) -> tf.Tensor:

        x = self.conv(x)  # (b,64,48,17)

        #out_1 = self.adptAP(x) # (b,1,1,17)
        out_1 = self.middle_path(x) # (b,1,1,17)
        out_2 = self.bottom_path(x) # (b,64,48,17)


        output = self.mul_1([out_1, out_2])
        output = self.mul_2([output, x])
        output = self.add_output([x, output])

        # output = Multiply(
        #     name= self.name + "b1_mul",dtype=tf.float32
        # )([out_1, out_2])

        # output = Multiply(
        #     name= self.name  + "b2_mul",dtype=tf.float32
        # )([output, x])

        # output = Add(
        #     name=self.name  +'RPM_out',  dtype=tf.float32
        # )([x, output])
        
        return output
#-----------------------------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------------------------
@MODELS.register_module()
class HeatmapBaseHead(BaseModule):
    VERSION = "1.0.0"
    r"""HeatmapBaseHead. classic heatmap head

    Topdown HeatmapBaseHead were consisted of 
    deconv layers(Conv2DTranspose) to upsample target shape and pointwise-conv1x1 to output target channels

    Top-down heatmap head introduced in `Simple Baselines`_ by Xiao et al
    (2018). The head is composed of a few deconvolutional layers followed by a
    convolutional layer to generate heatmaps from low-resolution feature maps.


    References:
        - [Inspired on implementation of 'HeatmapHead' @mmpose] 
         (https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/heads/heatmap_heads/heatmap_head.py) 
         

    Args:
        out_channels (int) : Number of channels in the output heatmap. Default to 17
        deconv_filters_list (List): The output channel number of each deconv layer 
            if len(deconv_filters_list) > 0, Defaults to [256, 256, 256].
        deconv_kernels_list (List):  The kernel size of each Conv2DTranspose, Defaults to [4,4,4]
        conv_kernel_szie (int):  The kernel size of each intermediate conv layer.  Defaults to ``None``
        conv_out_channels (int): the output channel number of each intermediate conv layer. 
            ``None`` means no intermediate conv layer between deconv layers 
            and the final conv layer. Defaults to ``None``
        activation (str):  activation. Defaults to 'relu'

    Note :
        input_transforms = [  
                dict(type="Reshape", target_shape=(16,12,-1),name="stage1_Reshape"),
                dict(type="Conv2D", filters=128, kernel_size=1,name="stage1_pwconv")
        ]
        

    Example:
    '''Python 
    
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input    
        x = Input(shape=(8*6, 256))
        out = HeatmapBaseHead(
                out_channels = 17,
                deconv_filters_list=[256, 256, 256], 
                deconv_kernels_list = [4,4,4],
                conv_kernel_szie = 3,
                conv_out_channels = 128,
                input_transforms = [  
                    dict(type="Reshape", target_shape=(8,6,-1),name="stage1_Reshape"),
                    dict(type="Conv2D", filters=128, kernel_size=1,name="stage1_pwconv")
                ], 
                activation ="relu", 
                name ='HeatmapBaseHead', 
                bn_epsilon = 1e-5,
                bn_momentum = 0.9,
        )(x)
        model = Model(x, out)
        model.summary(200)             
    """
    def __init__(self, 
            out_channels : int = 17,
            deconv_filters_list : List[int]=[256, 256, 256], 
            deconv_kernels_list : List[int]= [4,4,4],
            conv_kernel_szie : Optional[int] = None,
            conv_out_channels : Optional[int] = None,
            use_prm : bool = False,
            activation : str ="relu", 
            name : str='HeatmapBaseHead', 
            **kwargs): 
        
        super().__init__(
            name=name,  activation=activation,  **kwargs
        )
        assert len(deconv_filters_list)==len(deconv_kernels_list), \
        " len( deconv_filters_list) must be equal to len(deconv_kernels_list)" 

        self.out_channels = out_channels
        'CONV2D init'
        self.kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
        self.bias_init = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))

        self.num_deconv_layers = len(deconv_filters_list)
        self.deconv_filters_list = deconv_filters_list
        self.deconv_kernels_list = deconv_kernels_list
        
        self.conv_kernel_szie = conv_kernel_szie
        self.conv_out_channels = conv_out_channels

        self.use_prm = use_prm
       
        'input_transform'


    # def _make_deconv_layers(
    #     self,
    #     filters : int = 256, 
    #     kernel_size : int = 4,
    #     name : str = 'deconv_bn'
    # ):
    #     deconv_bn = tf.keras.Sequential(
    #         layers = [
    #             Conv2DTranspose(
    #                 filters = filters, 
    #                 kernel_size = kernel_size, 
    #                 strides = 2, 
    #                 padding="same",
    #                 use_bias=False,
    #                 name= 'TransConv'
    #             ) ,
    #             BatchNormalization(
    #                 epsilon=self.bn_epsilon,
    #                 momentum=self.bn_momentum,
    #                 name="bn"
    #             ),
    #             Activation(
    #                 self.act_name, 
    #                 name=self.act_name 
    #             )
    #         ] ,
    #         name = name  
    #     ) 
    #     return deconv_bn
    


    def build(self, input_shape):
        
        if hasattr(self,'input_transforms_map'):
            tf.print(self.input_transforms_map.layers)
        
        'classic heatmap head using Conv2DTranspose to upsample'  
        if self.num_deconv_layers > 0 :
            # self.conv_trans_list = [
            #     self._make_deconv_layers(
            #         filters  = deconv_filters, 
            #         kernel_size  = deconv_kernels,
            #         name = self.name +f'TransConvBn{i+1}'         
            #     )
            #     for i, (deconv_filters, deconv_kernels) in enumerate( zip(self.deconv_filters_list, self.deconv_kernels_list) )    
            # ]

            self.conv_trans_list = [ 
                Conv2DTranspose_BN(
                    filters = deconv_filters,
                    kernel_size = deconv_kernels,
                    strides=2,
                    use_bias = False,
                    use_bn = True,
                    activation = self.act_name,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    name =  self.name +f'TransConvBn{i+1}'   
                ) for i, (deconv_filters, deconv_kernels) in enumerate( zip(self.deconv_filters_list, self.deconv_kernels_list) )   
            ]

            # self.conv_trans_list = [
            #     Conv2DTranspose(
            #         filters = deconv_filters, 
            #         kernel_size = deconv_kernels, 
            #         strides = 2, 
            #         padding="same",
            #         use_bias=False,
            #         name= self.name +f'TransConv{i+1}'
            #     )
            #     for i, (deconv_filters, deconv_kernels) in enumerate( zip(self.deconv_filters_list, self.deconv_kernels_list) )    
            # ]



        if self.conv_out_channels and self.conv_kernel_szie:
            self.conv = Conv2D_BN(
                    filters = self.conv_out_channels ,
                    kernel_size= self.conv_kernel_szie,
                    strides=1,
                    use_bias = False,
                    use_bn = True,
                    activation = self.act_name,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    name =  self.name + 'Conv'
            )
                
        self.pwconv = Conv2D(
                filters=self.out_channels, 
                kernel_size = 1, 
                padding="same", 
                activation=None, 
                kernel_initializer=self.kernel_init,
                bias_initializer=self.bias_init,
                name =  self.name + 'pwConv_OUT',
                dtype="float32"
        )

        if self.use_prm :
            self.prm = PoseRefineMachine(            
                kernel_size = 3, 
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,   
                activation = self.act_name, 
                name=self.name+'RPM'
            )
            
    def call(self, x : tf.Tensor) -> tf.Tensor:

        if hasattr(self, 'conv_trans_list'):
            for conv_trans in self.conv_trans_list :
                x = conv_trans(x)

        if hasattr(self,'conv'):
           x = self.conv(x) 

        heatmap = self.pwconv(x)

        if hasattr(self, 'prm'):
            heatmap = self.prm(heatmap)

        return heatmap
    



#-----------------------------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------------------------
@MODELS.register_module()
class HeatmapSimpleHead(BaseModule):
    VERSION = '1.0.0'
    r"""HeatmapSimpleHead.

    TopdownHeatmapSimpleHead is to simply apply a upsample layer (UpSampling2D) to resize feature
 
 
    Required Data's Values dtype:
        - tf.Tensor
    Args:
        upsample_size (int) : Int, or tuple of 2 integers, same arg as tf.keras.layers.UpSampling2D()
                The upsampling factors for rows and columns. Defaults to (2,2)
        interpolation (str) : the interpolation method used by tf.keras.layers.UpSampling2D().  
                Supports "bilinear", "nearest", "bicubic", "area", "lanczos3", 
                "lanczos5", "gaussian", "mitchellcubic". Default to "bilinear"
        out_channels (int) : Number of channels in the output heatmap. Default to 17
        conv_kernel_szie (int):  The kernel size of each intermediate conv layer.  Defaults to ``None``
        conv_out_channels (int): the output channel number of each intermediate conv layer. 
            ``None`` means no intermediate conv layer between deconv layers 
            and the final conv layer. Defaults to ``None``
        activation(str) : activation used in intermediate conv layer , if conv_kernel_szie=mistr =conv_out_channels= None 
            this arg is invalid. Defaults to "relu", 
        name(str) : module's name.  Defaults to 'HeatmapSimpleHead', 
    Example:


            - simple_hm_head_cfg = dict(
                                        upsample_size= 4,
                                        interpolation = 'nearest'
                                        pre_conv_kernel = 3
                                        final_conv_kernel= 1,
                                        final_conv_out_channels = 17)
                     
    """
    def __init__(self,
            upsample_size : Optional[Union[Tuple[int],int]] =(2,2),
            interpolation : str = 'bilinear',
            out_channels : int = 17,
            conv_kernel_szie : Optional[int] =None, 
            conv_out_channels : Optional[int] = None,
            use_prm : bool = False,
            activation : str ="relu", 
            name='HeatmapSimpleHead', 
            **kwargs): 
        
        super().__init__(
            name=name,  activation=activation,  **kwargs
        )
        
        if upsample_size==None:
           upsample_size = (1,1)
        
        self.upsample_size = upsample_size
    
        self.interpolation = interpolation
        self.conv_kernel_szie = conv_kernel_szie
        self.conv_out_channels = conv_out_channels
        self.out_channels = out_channels
        self.use_prm = use_prm

        'ointwise CONV2D init'
        self.kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
        self.bias_init = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        

    def build(self, input_shape):
        
        if hasattr(self,'input_transforms_map'):
            tf.print(self.input_transforms_map.layers)

        if self.upsample_size!=(1,1):
            self.upsample2d = UpSampling2D(
                size = self.upsample_size, 
                interpolation=self.interpolation, 
                name=self.name + f'UpSampling2D'
            )

        if self.conv_out_channels and self.conv_kernel_szie:
            self.conv = Conv2D_BN(
                    filters = self.conv_out_channels ,
                    kernel_size= self.conv_kernel_szie,
                    strides=1,
                    use_bias = False,
                    use_bn = True,
                    activation = self.act_name,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    name =  self.name + 'ConvBn',
                    dtype="float32"
            ) 

        self.pwconv = Conv2D(
                filters=self.out_channels, 
                kernel_size = 1, 
                padding="same", 
                activation=None, 
                name = self.name + 'pwConv_OUT',
                kernel_initializer=self.kernel_init,
                bias_initializer=self.bias_init,
                dtype="float32")
        
        if self.use_prm :
            self.prm = PoseRefineMachine(            
                kernel_size = 3, 
                bn_epsilon = self.bn_epsilon,
                bn_momentum = self.bn_momentum,   
                activation = self.act_name, 
                name=self.name+'RPM'
            )
              
    def call(self, x : tf.Tensor) -> tf.Tensor:

        if hasattr(self,'upsample2d'):
            x = self.upsample2d(x)
        
        if hasattr(self,'conv'):
           x = self.conv(x) 

        heatmap = self.pwconv(x)

        if hasattr(self, 'prm'):
            heatmap = self.prm(heatmap)
        'simple heatmap'
        return heatmap






#-----------------------------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------------------------

@MODELS.register_module()
class TopdownHeatmapBaseHead(tf.Module):
    r"""TopdownHeatmapBaseHead.

    TopdownHeatmapBaseHead is consisted of (>=0) number of deconv layers(Conv2DTranspose) and pointwise-conv1x1
   
    Required Data's Values dtype:
        - tf.Tensor

    Args:
        deconv_filters_list (List): Number of filters in each Conv2DTranspose if len(List) > 0
        deconv_kernels_list (List): kernels used in each Conv2DTranspose if len(List) > 0
        pre_conv_kernel (int):  
        final_conv_kernel (int): kernels used in final Conv2D
        final_conv_out_channels (int):  Number of output channels

    Example:
              - input_feat : (16,12, 512) -> (64,48, 17)
                classical_hm_head_cfg = dict( econv_filters_list =[256, 256], 
                                            deconv_kernels_list = [4,4],
                                            final_conv_kernel= 1,
                                            final_conv_out_channels = 17)

                     
    """
    def __init__(self, 
            deconv_filters_list : List[int]=[256, 256, 256], 
            deconv_kernels_list : List[int]= [4,4, 4],
            final_conv_kernel : int = 1,
            final_conv_out_channels : int = 17,
            input_transforms : Optional[List[dict] ] = None, 
            pre_conv_kernel : Optional[int] = None,
            activation : str ="relu", 
            bn_epsilon : float= 1e-5,
            bn_momentum : float= 0.9,
            name : str='HeatmapBaseHead', 
            **kwargs): 
        
        super().__init__(name=name)
        assert len(deconv_filters_list)==len(deconv_kernels_list), \
        " len( deconv_filters_list) must be equal to len(deconv_kernels_list)" 
        'Batch Normalize cfg'

        'Batch Normalize cfg'
        #self.bn_m = bn_epsilon
        #self.bn_eps = bn_momentum
        self.bn_m = bn_momentum
        self.bn_eps = bn_epsilon   
        self.activation = activation
        'CONV2D init'
        self.kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
        self.bias_init = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))

        self.num_deconv_layers = len(deconv_filters_list)
        self.deconv_filters_list = deconv_filters_list
        self.deconv_kernels_list = deconv_kernels_list
        self.final_conv_kernel = final_conv_kernel
        self.final_conv_out_channels = final_conv_out_channels
        self.pre_conv_kernel = pre_conv_kernel

        'input_transform'
        self.input_transforms = input_transforms

        if isinstance(self.input_transforms, List) and self.input_transforms!=[]:
            self.input_transform_layers = tf.keras.Sequential(name="head/input_transforms")
            for cfg in self.input_transforms.copy() :
                self.input_transform_layers.add( getattr(layers,cfg.pop("type"))(**cfg))



    def __call__(self, input_feat : tf.Tensor) -> tf.Tensor:

        #assert isinstance(input_feat, tf.Tensor), "input_feat must be a tensor not list @TopdownHeatmapBaseHead"

        if isinstance(self.input_transforms, List) and self.input_transforms!=[]: 
            x = self.input_transform_layers(input_feat)
        else:
            x = input_feat

        # x = Reshape( 
        #     target_shape=(16,12,-1),name="head/input_transform/Reshape"
        # )(input_feat)



        'classic heatmap head using Conv2DTranspose to upsample'   
        if self.num_deconv_layers > 0 :
            for i, (deconv_filters, deconv_kernels) in enumerate( zip(self.deconv_filters_list, self.deconv_kernels_list) ): 
                x = layers.Conv2DTranspose(filters = deconv_filters, 
                                            kernel_size = deconv_kernels, 
                                            strides = 2, 
                                            padding="same",
                                            use_bias=False,
                                            name= f'Head/decoder-{i+1}/TransConv')(x)
                
                x = layers.BatchNormalization(epsilon=self.bn_eps,
                                            momentum=self.bn_m, 
                                            name=f'Head/decoder-{i+1}/bn')(x)
                
                x = layers.Activation( self.activation , name= f'Head/decoder-{i+1}/{self.activation}')(x)

        if self.pre_conv_kernel is not None :
            x = Conv2D_BN(filters = x.shape[-1],
                        kernel_size=self.pre_conv_kernel,
                        strides=1,
                        use_bias = False,
                        use_bn = True,
                        activation = self.activation,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        name = 'Head_PreConvBN')(x)
                   
        'heatmap_head'
        final_layer_name = "Head/pwConv_OUT"if self.final_conv_kernel==1 else f"Head/Conv{self.final_conv_kernel}x{self.final_conv_kernel}_OUT" 
        heatmap = layers.Conv2D(filters=self.final_conv_out_channels, 
                                kernel_size = self.final_conv_kernel, 
                                padding="same", 
                                activation=None, 
                                name = final_layer_name ,
                                kernel_initializer=self.kernel_init,
                                bias_initializer=self.bias_init,
                                dtype="float32")(x)
        
        return heatmap
    
    def get_config(self):
        config = super().get_config()
        
        config.update(
            {
                "num_deconv_layers" : self.num_deconv_layers, 
                "deconv_filters" : self.deconv_filters_list,  
                "deconv_kernels" : self.deconv_kernels_list,
                "pre_conv_kernel" : self.pre_conv_kernel,
                "final_conv_kernel" : self.final_conv_kernel,
                "final_conv_out_channels" : self.final_conv_out_channels,

            }
        )
        return config   

    @classmethod
    def from_config(cls, config):
        return cls(**config)    
    

# @MODELS.register_module()
# class TopdownHeatmapSimpleHead(tf.Module):
#     r"""TopdownHeatmapSimpleHead.

#     TopdownHeatmapSimpleHead is to simply apply a upsample layer (UpSampling2D) to resize feature
 
 
#     Required Data's Values dtype:
#         - tf.Tensor
#     Args:
#         upsample_size (int) :  same arg as tf.keras.layers.UpSampling2D( size=(2, 2), interpolation='nearest)
#         interpolation (str) : same arg as tf.keras.layers.UpSampling2D( size=(2, 2), interpolation='nearest)
#         final_conv_kernel (int): kernels used in final Conv2D
#         final_conv_out_channels (int):  Number of output channels

#     Example:


#             - simple_hm_head_cfg = dict(
#                                         upsample_size= 4,
#                                         interpolation = 'nearest'
#                                         pre_conv_kernel = 3
#                                         final_conv_kernel= 1,
#                                         final_conv_out_channels = 17)
                     
#     """
#     def __init__(self,
#             upsample_size : Union[Tuple[int],List[int]] =(2,2),
#             interpolation : str = 'nearest',
#             final_conv_kernel : int = 1,
#             final_conv_out_channels : int = 17,
#             pre_conv_kernel : Optional[int] =None, 
#             bn_epsilon : float= 1e-5,
#             bn_momentum : float= 0.9,
#             activation : str ="relu", 
#             input_transforms : Optional[List[dict] ] = None, 
#             name=None, 
#             **kwargs): 
        
#         super().__init__(name=name)
#         ''
#         if isinstance(upsample_size, (list, tuple)):
#             self.upsample_size = upsample_size
#         else :
#             self.upsample_size = (upsample_size, upsample_size)
    
#         self.upsample_method = interpolation

#         'Batch Normalize cfg'
#         self.bn_m = bn_epsilon
#         self.bn_eps = bn_momentum
#         self.activation = activation
#         'CONV2D init'
#         self.kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
#         self.bias_init = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        

#         'input_transform'
#         self.input_transforms = input_transforms

#         if isinstance(self.input_transforms, List) and self.input_transforms!=[]:
#             self.input_transform_layers = tf.keras.Sequential(name="head/input_transforms")
#             for cfg in self.input_transforms.copy() :
#                 self.input_transform_layers.add( getattr(layers,cfg.pop("type"))(**cfg))

#         self.pre_conv_kernel = pre_conv_kernel
#         self.final_conv_kernel = final_conv_kernel
#         self.final_conv_out_channels = final_conv_out_channels

        


#     def __call__(self, input_feat : tf.Tensor) -> tf.Tensor:

#         #assert isinstance(input_feat, tf.Tensor), "input_feat must be a tensor not list @TopdownHeatmapBaseHead"

#         if isinstance(self.input_transforms, List) and self.input_transforms!=[]: 
#             input_feat = self.input_transform_layers(input_feat)
 
#         'simple heatmap'
#         if self.upsample_size!=(1,1):
#             x = layers.UpSampling2D(size = self.upsample_size, 
#                             interpolation=self.upsample_method, 
#                             name=f'Head_UpSampling2D')(input_feat)
#         else:
#             x = input_feat

#         if self.pre_conv_kernel is not None :
#             x = Conv2D_BN(filters = x.shape[-1],
#                         kernel_size=self.pre_conv_kernel,
#                         strides=1,
#                         use_bias = False,
#                         use_bn = True,
#                         activation = self.activation,
#                         bn_epsilon = self.bn_epsilon,
#                         bn_momentum = self.bn_momentum,
#                         name = self.name + 'Head_PreConvBN')(x)
                
#         'heatmap_head'
#         final_layer_name = "Head_pwConv_OUT"if self.final_conv_kernel==1 else f"Head_Conv{self.final_conv_kernel}x{self.final_conv_kernel}_OUT" 
#         heatmap = layers.Conv2D(filters=self.final_conv_out_channels, 
#                                 kernel_size = self.final_conv_kernel, 
#                                 padding="same", 
#                                 activation=None, 
#                                 name = final_layer_name ,
#                                 kernel_initializer=self.kernel_init,
#                                 bias_initializer=self.bias_init,
#                                 dtype="float32")(x)
        
#         return heatmap
    
#     def get_config(self):
#         config = super().get_config()
        
#         config.update(
#             {
#                 "upsample_size" : self.upsample_size , 
#                 "upsample_method" : self.upsample_method,
#                 "pre_conv_kernel" : self.pre_conv_kernel,
#                 "final_conv_kernel" : self.final_conv_kernel,
#                 "final_conv_out_channels" : self.final_conv_out_channels,
                

#             }
#         )
#         return config   

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)    
    


# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input

# x =  Input(shape=(16,12,128))


# out = TopdownHeatmapSimpleHead( 
#             upsample_size= 4,
#             interpolation = 'nearest',
#             pre_conv_kernel =None, 
#             final_conv_out_channels = 17,
#             final_conv_kernel = 1,
#             activation ="relu", 
#             bn_epsilon = 1e-5,
#             bn_momentum = 0.9,
#             name='SimpleHMHead')(x)
# model = Model(x, out)
# model.summary(150)        
