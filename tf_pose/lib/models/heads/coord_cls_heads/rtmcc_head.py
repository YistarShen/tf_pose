from typing import Tuple
from tensorflow.keras.layers import Dense, Concatenate, Reshape, Permute
from lib.models.modules import BaseModule
from lib.layers import Conv2D_BN, ScaleNormLayer, GatedAttentionUnit
from lib.Registers import MODELS
#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
@MODELS.register_module()
class RTMCCHead(BaseModule):
    VERSION = '1.0.0'
    r""" RTMCCHead
    Top-down head introduced in RTMPose (2023). The head is composed of a
    large-kernel convolutional layer, a fully-connected layer and a Gated
    Attention Unit to generate 1d representation from low-resolution feature
    maps.

    
    Args:
        simcc_split_ratio (float) :  Split ratio of pixels. Default: 2.0.
        input_size_hw (Tuple[int]) : Size of input image in shape [h, w] , Defaults to  (256,192).
        hidden_dims (int) : default to 256.
        gau_expansion_factor (float) :  default to 2.0
        gau_att_dims (int) :  default to 128
        gau_drop_rate (float) :  default to .0
        ln_epsilon (float) : epsilon of layer normalization , defaults to 1e-5.
        conv_out_channels (int) : Number of channels in the output heatmap that means num_keypoint, defaults to 17.
        conv_kernel_size (int) : Kernel size of the convolutional layer, defaults to 7.
        output_concatenate (bool) : defaults to False.

    Note :
       -  

    References:
         - [Based on implementation of 'RTMCCHead' @mmpose ]
            (https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/heads/coord_cls_heads/rtmcc_head.py)

    Example:
        -------------------------------------------------------
        '''Python
        from lib.Registers import MODELS
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        x = Input(shape=(8, 6, 1024))
        out = RTMCCHead(    
                simcc_split_ratio = 2.0,   
                input_size_hw = (256,192),       
                hidden_dims = 256,
                gau_expansion_factor =2.,
                gau_att_dims = 128,
                ln_epsilon = 1e-5,
                conv_out_channels = 17, 
                conv_kernel_size = 7,
                output_concatenate = False,
                name='RTMCCHead'
        )(x)
        model = Model(x, out)
        model.summary(200)
    
    """
    def __init__(self,
        simcc_split_ratio : float = 2.,
        input_size_hw : Tuple[int] = (256,192) ,    
        hidden_dims : int = 256,
        gau_expansion_factor : float =2.,
        gau_att_dims : int = 128,
        gau_drop_rate : float = 0.,
        ln_epsilon : float= 1e-5,
        conv_out_channels : int = 17, 
        conv_kernel_size: int =7,
        output_concatenate : bool = False,
        name : str='RTMCCHead', 
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.H = int( input_size_hw[0]*simcc_split_ratio)
        self.W = int( input_size_hw[1]*simcc_split_ratio)
        self.hidden_dims = hidden_dims
        self.gau_expansion_factor = gau_expansion_factor
        self.gau_att_dims = gau_att_dims
        self.gau_drop_rate = gau_drop_rate
        self.conv_out_channels = conv_out_channels
        self.conv_kernel = conv_kernel_size   
        self.ln_epsilon = ln_epsilon
        self.wrap_outputs = output_concatenate

        
    def build(self, input_shape):

        self.n, self.h, self.w, self.c = input_shape.as_list()


    def call(self, inputs):

        x = Conv2D_BN(
            filters = self.conv_out_channels,
            kernel_size = self.conv_kernel,
            strides=1,
            use_bn = False,
            activation = None,
            deploy = None,
            name = self.name+f'Conv{self.conv_kernel}x{self.conv_kernel}'
        )(inputs)   #(b, h, w, 17)

        x = Reshape(
            (-1,self.conv_out_channels), name=self.name+"Reshape"
        )(x)    #(b, h*w, 17)
        x = Permute(
            (2, 1),name=self.name+"Permute"
        )(x) #(b,  17, h*w)
        x = ScaleNormLayer(
            ln_epsilon = self.ln_epsilon, name=self.name+"ScaleNormLayer"
        )(x)  #(b,  17, h*w)
        x = Dense(
            units=self.hidden_dims, use_bias=False, name=self.name+"mlp"
        )(x)   #(b,  17, h*w)) =>  #(b,  17, 256)
        x = GatedAttentionUnit(   
            out_token_dims = self.hidden_dims,
            query_key_dims =self.gau_att_dims,
            expansion_factor = self.gau_expansion_factor,
            norm_type = 'scale_norm',
            dropout_rate = self.gau_drop_rate,
            ln_epsilon = self.ln_epsilon,
            use_bias = False,
            use_laplace_attn = False,
            use_rel_pos_bias = False,
            use_rope = False,
            name=self.name+"GAU"
        )(x)     #(b,  17, 256) =>  #(b,  17, 256)
        cls_x = Dense(
            units=self.W, use_bias=False, 
            name=self.name+"dens_cls_x",dtype="float32"
        )(x) #(b,,17,192*2)
        cls_y =Dense(
            units=self.H, use_bias=False, 
            name=self.name+"dens_cls_y", dtype="float32"
        )(x) #(b,,17,256*2)

        if self.wrap_outputs:
            output = Concatenate(
                axis=-1, name = self.name+"cls_xy_output",  dtype="float32"
            )([cls_x,cls_y]) 
        else:
            output =  [cls_x, cls_y]

        return output
    




# class RTMCCHead(BaseModule):
#     def __init__(self,
#         out_dims_hw = (256*2,192*2),       
#         hidden_dims = 256,
#         gau_expansion_factor =2.,
#         gau_att_dims = 128,
#         ln_epsilon : float= 1e-5,
#         conv_out_channels = 17, 
#         conv_kernel_size=7,
#         output_concatenate=False,
#         name : str='RTMCCHead', 
#         **kwargs
#     ):
#         super().__init__(name, **kwargs)
#         self.H = int( out_dims_hw[0])
#         self.W = int( out_dims_hw[1])
#         self.hidden_dims = hidden_dims
#         self.gau_expansion_factor = gau_expansion_factor
#         self.gau_att_dims = gau_att_dims
#         self.conv_out_channels = conv_out_channels
#         self.conv_kernel = conv_kernel_size   
#         self.wrap_outputs = output_concatenate
#         self.ln_epsilon = ln_epsilon
        
#     def build(self, input_shape):

#         self.n, self.h, self.w, self.c = input_shape.as_list()

#         self.conv = Conv2D_BN(
#                 filters = self.conv_out_channels,
#                 kernel_size = self.conv_kernel,
#                 strides=1,
#                 use_bn = False,
#                 activation = None,
#                 deploy = None,
#                 name = self.name+f'Conv_{self.conv_kernel}x{self.conv_kernel}'
#         ) #(b, h, w, 17)

#         self.seq = tf.keras.Sequential(
#             layers = [
#                 Reshape(
#                     (-1,self.conv_out_channels), name=self.name+"Reshape"
#                 ), #(b, h*w, 17)
#                 Permute(
#                     (2, 1),name=self.name+"permute"
#                 ), #(b,  17, h*w)
#                 ScaleNormLayer(
#                     ln_epsilon = self.ln_epsilon, name=self.name+"ScaleNormLayer"
#                 ),  #(b,  17, h*w)
#                 Dense(
#                     units=self.hidden_dims, use_bias=False, name=self.name+"mlp"
#                 ),  #(b,  17, 256)
#                 GatedAttentionUnit(   
#                     out_token_dims = self.hidden_dims,
#                     query_key_dims =self.gau_att_dims,
#                     expansion_factor = self.gau_expansion_factor,
#                     norm_type = 'scale_norm',
#                     dropout_rate = 0.,
#                     ln_epsilon = self.ln_epsilon,
#                     use_bias = False,
#                     use_laplace_attn = False,
#                     use_rel_pos_bias = False,
#                     use_rope = False,
#                     name=self.name+"GAU"
#                 )   #(b,  17, 256) =>  #(b,  17, 256)
#             ] ,
#             name = self.name+'RTMCCBlock'    
#         )
        
#         self.cls_x = Dense(
#             units=self.W, use_bias=False, 
#             name=self.name+"dens_cls_x",dtype=tf.float32
#         )
#         self.cls_y =Dense(
#             units=self.H, use_bias=False, 
#             name=self.name+"dens_cls_y", dtype=tf.float32
#         ) 

#         if self.wrap_outputs :
#             self.concat = tf.keras.layers.Concatenate(
#                 axis=-1, dtype=self.out_dtype, name=self.name+"_out_xy"
#             )

#     def call(self, inputs):

#         x = self.conv(inputs) #(b,8,6,1024) => (b,8,6,17)
#         x = self.seq(x) #(b,8,6,17) => (b,8*6,17) =>  (b,17, 8*6) =>  (b,17, hidden_dims) => (b,17, hidden_dims) 

#         pred_x = self.cls_x(x) #(b,,17,192*2)
#         pred_y = self.cls_y(x) #(b,,17,256*2)

#         # if self.wrap_outputs :
#         #     outputs = tf.keras.layers.Concatenate(
#         #             axis=-1, dtype=self.out_dtype, name=self.name+"_out_xy"
#         #     )([pred_x,pred_y])
#         # else:
#         #     outputs = [pred_x,pred_y] 

#         return self.concat([pred_x,pred_y])  if hasattr(self,'concat') else [pred_x, pred_y]
