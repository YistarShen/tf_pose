from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from tensorflow.keras.layers import Concatenate, Add,  Activation, Resizing, MaxPooling2D
import tensorflow as tf
from lib.models.backbones.base_backbone import BaseBackbone
from lib.models.modules import ResidualStepsBlock
from lib.layers import Conv2D_BN
from lib.Registers import MODELS

BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.97
#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
def RSN_DownsampleModule(
    num_blocks=[2, 2, 2, 2],
    num_units : int= 4,
    num_steps : int = 4,
    expand_times  : int = 26,
    res_top_channels : int = 64, 
    kernel_size : int = 3,
    use_depthwise : bool = False,
    use_skip : bool = True,
    activation : str='relu',
    deploy : Optional[bool] = None,
    name : str ="RSN_DS",
    **kwargs
):
    r""" 
    
    Example:
    -------------------------------------------------------
        '''Python
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        x = Input(shape=(64,48,64))
        skip1 = [Input(shape=(64,48,64)), Input(shape=(32,24,128)), Input(shape=(16,12,256)), Input(shape=(8,6,512))]
        skip2 = [Input(shape=(64,48,64)), Input(shape=(32,24,128)), Input(shape=(16,12,256)), Input(shape=(8,6,512))]
        inputs = [x,skip1,skip2]

        out = RSN_DownsampleModule(         
                        num_blocks=[2, 2, 2, 2],
                        num_units = 4,
                        num_steps= 4,
                        expand_times = 26,
                        res_top_channels  = 64, 
                        kernel_size  = 3,
                        use_depthwise = False,
                        use_skip  = True,
                        activation ='relu',
                        deploy = False,
                        bn_epsilon = 1e-5,
                        bn_momentum = 0.9,
                        name  ="downsample"
        )(x,skip1,skip2)
        model = Model((x,skip1,skip2), out)
    
    """
    name = name + "_"
    def apply(x, skip1=None, skip2=None):

        if skip1!=None:
            if not all([ True 
                if (a.shape[1]==b.shape[1]*2 and  a.shape[2]==b.shape[2]*2) 
                else False 
                for a, b in zip(skip1[:-1],skip1[1:])]
            ):

                raise ValueError(
                    f"the shapes of skip1 must be in descending order ,i.e. from P2->P5"    
                    f"but got {[ X.shape for X in (skip1)]} "   
                ) 
            
            
        if skip2!=None:
            if not all([ True 
                if (a.shape[1]==b.shape[1]*2 and  a.shape[2]==b.shape[2]*2) 
                else False 
                for a, b in zip(skip2[:-1],skip2[1:])]
            ):

                raise ValueError(
                    f"the shapes of skip2 must be in descending order ,i.e. from P2->P5"    
                    f"but got {[ X.shape for X in (skip2)]} "   
                ) 
            
        
        outputs = []
        for unit_ith in range(num_units):
            for block_ith in range(num_blocks[unit_ith]):
    
                strides = 2 if (unit_ith!=0 and block_ith==0)  else 1
                #downsample_channels[unit_ith], res_top_channels*(2**(i))
                x = ResidualStepsBlock(
                        out_channels = res_top_channels*(2**(unit_ith)),
                        expand_times = expand_times,
                        num_steps = num_steps,
                        strides = strides,
                        res_top_channels = res_top_channels, 
                        kernel_size = kernel_size,
                        use_depthwise = use_depthwise,
                        activation = activation,
                        deploy = deploy,
                        name  = name+f'P{unit_ith+2}_RSB{block_ith+1}',
                        **kwargs
                )(x)

            if use_skip :
                x = Add(
                    name=name+f'P{unit_ith+2}_skipAdd'
                )([x, skip1[unit_ith], skip2[unit_ith]])

            outputs.append(x)
        return outputs
    
    return apply


#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
def  RSN_UpsampleUnit(
    unit_channels : int= 256,
    out_channels : int = 64,
    interpolation : str = 'bilinear',
    gen_skip : bool = True,
    gen_cross_conv : bool = False,
    activation : str='relu',
    deploy : bool = None,
    name : str ="UpsampleUnit", 
    **kwargs
): 
    r""" 
    
    Example:
    -------------------------------------------------------
        '''Python

        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        x = Input(shape=(64,48,64))
        up_to_x = Input(shape=(32,24,128))
        outputs = RSN_UpsampleUnit(
                        unit_channels = 256,
                        out_channels  = 64,
                        interpolation  = 'bilinear',
                        gen_skip  = True,
                        gen_cross_conv  = False,
                        activation ='relu',
                        name ="UpsampleUnit",                           
                        bn_epsilon = 1e-5,
                        bn_momentum = 0.9
        )(x,up_to_x)
        model = Model((x,up_to_x), outputs)
    """
    name = name +"_"
    def apply(x, up_to_x=None):
        out = Conv2D_BN(
            filters = unit_channels,
            kernel_size=1,
            strides=1,
            activation = None,
            deploy=deploy,
            name = name+'prev_conv',
            **kwargs
        )(x)

        if up_to_x!= None :
            up_to_x = Resizing(
                        x.shape[1], 
                        x.shape[2], 
                        interpolation=interpolation,  
                        name=name+'Resize'
            )(up_to_x)

            up_to_x = Conv2D_BN(
                        filters = unit_channels,
                        kernel_size=1,
                        strides=1,
                        activation = None,
                        name =name+'up_conv',
                        **kwargs
            )(up_to_x)

            out = Add(name=name + 'fuse_Add')([out,up_to_x])

        out = Activation(
            'relu',name=name+'out_relu'
        )(out)
        #-------------------------------------------------------------------
        skip1 = None
        skip2 = None
        if gen_skip:
            skip1 = Conv2D_BN(
                        filters = x.shape[-1],
                        kernel_size=1,
                        strides=1,
                        activation = activation,
                        deploy=deploy,
                        name = name+f'skip1_conv',
                        **kwargs
            )(x)
            
            skip2 = Conv2D_BN(
                        filters = x.shape[-1],
                        kernel_size=1,
                        strides=1,
                        activation = activation,
                        deploy=deploy,
                        name = name+f'skip2_conv',
                        **kwargs
            )(out)
        #-------------------------------------------------------------------
        cross_conv = None
        if gen_cross_conv:
            cross_conv = Conv2D_BN(
                            filters = out_channels,
                            kernel_size=1,
                            strides=1,
                            activation = activation,
                            deploy= deploy,
                            name = name+f'cross_conv',
                            **kwargs
            )(out) 
        return  out, skip1, skip2, cross_conv
    return apply


#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
def  RSN_UpsampleModule(
    unit_channels : int= 256,
    out_channels : int = 64,
    num_units : int = 4,
    interpolation : str = 'bilinear',
    gen_skip : bool = True,
    gen_cross_conv : bool = False,
    activation : str='relu',
    deploy : Optional[bool] = None,
    name : str ="UpsampleUnit",
    **kwargs
):
    
    r""" 
    
    Example:
    -------------------------------------------------------
        '''Python

        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        xs = [Input(shape=(64,48,64)), Input(shape=(32,24,128)), Input(shape=(16,12,256)), Input(shape=(8,6,512))]

        outputs = RSN_UpsampleModule(
            unit_channels = 256,
            out_channels  = 64,
            num_units = 4,
            interpolation  = 'bilinear',
            gen_skip  = True,
            gen_cross_conv = False,
            activation ='relu',
            name  ="Module",  
            bn_epsilon = 1e-5,
            bn_momentum = 0.9
        )(xs[::-1])  
        model = Model(xs[::-1], outputs)
    """
    name = name +"_"
    def apply(xs):
        if len(xs)!= num_units:
            raise ValueError(
            f"len(xs) must be equal to num_units={num_units} "    
            f", but got {len(xs)} "   
            )
        
        if not all([ True 
                if (a.shape[1]*2==b.shape[1] and  a.shape[2]*2==b.shape[2]) 
                else False 
                for a, b in zip(xs[:-1],xs[1:])]
            ):

            raise ValueError(
                f"the shapes of input xs(list) must be in ascending order ,i.e. from P5->P2"    
                f"but got {[ X.shape for X in (xs)]} "   
            )   
        out = list()
        skip1 = list()
        skip2 = list()
        cross_conv = None
        for i in range(num_units):
            unit = RSN_UpsampleUnit(
                        unit_channels = unit_channels,
                        out_channels = out_channels,
                        interpolation = interpolation,
                        gen_skip = gen_skip ,
                        gen_cross_conv  =  gen_cross_conv if num_units-1==i else False,
                        activation =activation,
                        deploy = deploy,
                        name =name+f"P{5-i}",
                        **kwargs
            )

            if i == 0:
                outi, skip1_i, skip2_i, _ = unit(xs[i],None)
            elif i == num_units - 1:
                outi, skip1_i, skip2_i, cross_conv = unit(xs[i], out[i - 1])
            else:
                outi, skip1_i, skip2_i, _ = unit(xs[i], out[i - 1])

            out.append(outi)
            skip1.append(skip1_i)
            skip2.append(skip2_i)

        return out, skip1[::-1], skip2[::-1], cross_conv
    
    return apply


#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
def RSN_SingleSatage(
    use_skip=False,
    gen_skip=False,
    gen_cross_conv=False,
    unit_channels=256,
    num_units=4,
    num_steps=4,
    num_blocks=[2, 2, 2, 2],
    in_channels=64,
    expand_times=26,   
    activation  : str = 'relu',
    deploy  : Optional[bool] = None,
    name : str = "Stage1_RSN",
    **kwargs
):
    r""" 
    
    Example:
    -------------------------------------------------------
        '''Python

        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        x =Input(shape=(64,48,64))
        skip1 = [Input(shape=(64,48,64)), Input(shape=(32,24,128)), Input(shape=(16,12,256)), Input(shape=(8,6,512))]
        skip2 = [Input(shape=(64,48,64)), Input(shape=(32,24,128)), Input(shape=(16,12,256)), Input(shape=(8,6,512))]

        outputs = RSN_SingleSatage(
            use_skip=True,
            gen_skip=True,
            gen_cross_conv=True,
            unit_channels=256,
            num_units=4,
            num_steps=4,
            num_blocks=[2, 2, 2, 2],
            in_channels=64,
            expand_times=26,   
            activation  = 'relu',
            deploy  = None,
            name = "Stage1_RSN",
            bn_epsilon = 1e-5,
            bn_momentum = 0.9
        )(x, skip1=skip1, skip2=skip2)  

        model = Model((x, skip1, skip2), outputs)
    """    
    name = name + "_"
    def apply( x, skip1=None, skip2=None):

        mid = RSN_DownsampleModule(         
                num_blocks=num_blocks,
                num_units = num_units,
                num_steps= num_steps,
                expand_times = expand_times,
                res_top_channels  = in_channels, 
                kernel_size  = 3,
                use_depthwise = False,
                use_skip  = use_skip,
                activation =activation,
                deploy = deploy,
                name  = name+"Down",
                **kwargs
        )(x,skip1,skip2)

        outputs, skip1, skip2, cross_conv  = RSN_UpsampleModule(
                unit_channels = unit_channels,
                out_channels  = in_channels,
                num_units = num_units,
                interpolation  = 'bilinear',
                gen_skip  = gen_skip,
                gen_cross_conv = gen_cross_conv,
                activation =activation,
                deploy = deploy,
                name  = name + "Up",  
                **kwargs
        )(mid[::-1])   

        return outputs, skip1, skip2, cross_conv
    
    return apply

#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
@MODELS.register_module()
class ResidualStepsNetwork(BaseBackbone):
    
    VERSION = '1.0.0'

    r"""ResidualStepsNetwork(RSN)

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
        model_input_shape : Tuple[int,int]=(256,192),
        num_stages: int = 4,
        num_units : int =4,
        num_blocks : List[int]= [2, 2, 2, 2],
        num_steps=4,
        res_top_channels : int = 64,
        expand_times : int = 26,
        use_depthwise: bool = False,
        data_preprocessor: dict = None,
        name =  'RSN',
        *args, 
        **kwargs):


        self.num_stages = num_stages
        self.num_units = num_units
        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.res_top_channels = res_top_channels
        self.expand_times = expand_times
        self.use_depthwise = use_depthwise

        super().__init__(input_size =  (*model_input_shape,3),
                        data_preprocessor = data_preprocessor,
                        name = name, **kwargs)

    def call(self,  x:tf.Tensor)->tf.Tensor:


        x = Conv2D_BN(
            filters = self.res_top_channels,
            kernel_size=7,
            strides=2,
            bn_epsilon = self.bn_epsilon,
            bn_momentum = self.bn_momentum,
            activation = self.act_name,
            deploy  = self.deploy,
            name = 'Stem_P1_Conv',

        )(x)

        x = MaxPooling2D(
            pool_size=3, strides=2, padding='same', name='Stem_P2_MaxPooling2D'
        )(x)

        skip1 = None
        skip2 = None
        RSN_stage_outputs_list = []

        for ith_stage in range(self.num_stages):
            'no skip layer input for downsample module in the first stage  '
            use_skip = True if ith_stage != 0 else False
            'no skip layer generation for upsample module in the last stage '
            gen_skip = True if ith_stage != self.num_stages - 1 else False
            'no cross_conv layer generation for upsample module in the last stage '
            gen_cross_conv = True if ith_stage != self.num_stages - 1 else False 


            outputs = RSN_SingleSatage(
                    use_skip=use_skip,
                    gen_skip=gen_skip,
                    gen_cross_conv=gen_cross_conv,
                    unit_channels=256,
                    num_units=self.num_units,
                    num_steps=self.num_steps,
                    num_blocks=self.num_blocks,
                    in_channels=self.res_top_channels,
                    expand_times=self.expand_times ,   
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy  = self.deploy,
                    name = f"Stage{ith_stage}_RSN",
            )(x, skip1=skip1, skip2=skip2)   

            #ascending order : skip1,skip2
            feats_out, skip1, skip2, x  = outputs
            RSN_stage_outputs_list.append(feats_out[::-1])
        
        #RSN_stage_outputs_list.append(x)
        return  RSN_stage_outputs_list