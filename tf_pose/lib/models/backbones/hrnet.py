from typing import Optional,Tuple,List
import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Layer, Add, Activation

from .base_backbone import BaseBackbone
from lib.Registers import MODELS
from lib.layers import Conv2D_BN
from lib.models.modules import BasicResModule, ResBottleneck

BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.97
#----------------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------------
def  HR_FuseBlock(
    num_out_branches : int,
    interpolation : str = 'bilinear',
    activation : str='relu',
    deploy : bool = None,
    name : str ="HR_FuseModule", 
    **kwargs
): 
    name = name + "_"
    def apply(xs):
        if not isinstance(xs,list):
            raise TypeError(
            f"input xs must be List[Tensor], but got {type(xs)} @{HR_FuseBlock.__name__}"
            )
        
        if len(xs) == 1:
            return xs
         
        fuse_outputs = []
        for i in range(num_out_branches):
            to_be_fused = []
            for j in range(len(xs)):
                x = xs[j]
                if j > i:
                    x = Conv2D_BN(
                        filters = xs[i].shape[-1],
                        kernel_size=1,
                        strides=1,
                        activation = None,
                        deploy=deploy,
                        name = name+f'B{i+1}_P{j+2}_pwConvBn',
                        **kwargs
                    )(x)

                    x = UpSampling2D(
                        size=2 ** (j - i), 
                        interpolation=interpolation,
                        name=name + f'B{i+1}_P{j+2}_UpSample2D'
                    )(x)
                elif j < i:
                    for k in range(i - j):
                        if k == i - j - 1:
                            x = Conv2D_BN(
                                filters = xs[i].shape[-1],
                                kernel_size=3,
                                strides=2,
                                activation = None,
                                deploy=deploy,
                                name = name+f'B{i+1}_P{j+2}_ConvBn{k+1}',
                                **kwargs
                            )(x) 
                        else:
                            x = Conv2D_BN(
                                filters = xs[j].shape[-1],
                                kernel_size=3,
                                strides=2,
                                activation = activation,
                                deploy=deploy,
                                name = name+f'B{i+1}_P{j+2}_ConvBn{k+1}',
                                **kwargs
                            )(x) 
                else:
                    x = Layer(name=name+f'B{i+1}_P{j+2}_Identity')(x)

                to_be_fused.append(x)      

            x = Add(name=name + f'B{i+1}_FuseAdd')(to_be_fused)
            x = Activation('relu', name=name + f'B{i+1}_relu_out')(x)  
            fuse_outputs.append(x)

        return fuse_outputs
    
    return apply

#----------------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------------
def  HR_TransitionBlock(
    output_channels : list,
    activation : str='relu',
    deploy : bool = None,
    name : str ="TransModule", 
    **kwargs
): 
    name = name  + '_'
    def apply(xs):
        num_branchs_pre = len(xs)
        num_bbranchs_cur = len(output_channels)
        xs_out = []
        for i in range(num_bbranchs_cur):
            if i < num_branchs_pre:
                x = xs[i]
                if x.shape[-1] != output_channels[i]:
                    x = Conv2D_BN(
                        filters = output_channels[i],
                        kernel_size=3,
                        strides=1,
                        activation = activation,
                        deploy=deploy,
                        name = name+f'B{i+1}_ConvBn',
                        **kwargs
                    )(x)
                else:
                    x = Layer(
                        name=name+f'B{i+1}_Identity'
                    )(x)
            else:
                x = xs[-1]
                x = Conv2D_BN(
                    filters = output_channels[i],
                    kernel_size=3,
                    strides=2,
                    activation = activation,
                    deploy=deploy,
                    name = name+f'B{i+1}_dsConvBn',
                    **kwargs
                )(x)
            xs_out.append(x)
        return xs_out
    return apply

#----------------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------------
def  HR_BaseBlock(
    branches_channels : list,
    num_blocks : int ,
    use_bottleneck_blocks : bool = False,
    multiscale_output : bool=True,
    interpolation : str = 'bilinear',
    activation : str='relu',
    deploy : bool = None,
    psa_type :Optional[str]=None,
    name : str ="HR_Module", 
    **kwargs
): 
    name = name +"_"
    num_out_branches =  len(branches_channels) if multiscale_output else 1 

    def apply(xs):

        assert isinstance(xs, list), \
        "input xs must be 'list' type "

        if not len(xs)==len(branches_channels) :
            raise ValueError(
                "len(xs) must be len(branches_channels), \n"
                f"bot got len(xs)={len(xs)} and len(branches_channels)={len(branches_channels)}"
            )
      
        xs_outputs = []
        for j in range (len(xs)):
            x = xs[j]
            for k in range(0,num_blocks):
                #block_name =  name + f'B{j+1}_Bottleneck{k+1}' if use_bottleneck_blocks else f'B{j+1}_ResBlock{k+1}'
                if use_bottleneck_blocks :
                    block_name =  name + f'B{j+1}_Bottleneck{k+1}'
                    x = ResBottleneck(
                        out_channels = branches_channels[j],
                        hidden_channel_ratio = 0.25,
                        strides = 1,
                        activation = activation,
                        deploy = deploy,
                        name  = block_name,
                        **kwargs
                    )(x)
                else:
                    block_name =  name + f'B{j+1}_ResBlock{k+1}'
                    x = BasicResModule(
                        out_channels = branches_channels[j],
                        strides = 1,
                        activation = activation,
                        psa_type  = psa_type,
                        deploy = deploy,
                        name  = block_name,
                        **kwargs
                    )(x)  


                # x = ResidualBlock(
                #     out_channels = branches_channels[j],
                #     hidden_channel_ratio = 0.25 if use_bottleneck_blocks else 1,
                #     is_bottleneck = use_bottleneck_blocks,
                #     strides = 1,
                #     activation = activation,
                #     deploy = deploy,
                #     name  = name + block_name,
                #     **kwargs
                # )(x)
            xs_outputs.append(x)
        
        xs_outputs = HR_FuseBlock(
            num_out_branches = num_out_branches,
            interpolation = interpolation,
            activation =activation,
            deploy = deploy,
            name = name +f'FuseBlock', 
            **kwargs
        )(xs_outputs)

        return xs_outputs
    
    return apply


#----------------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------------
@MODELS.register_module()
class  HRNet(BaseBackbone):
    VERSION = '1.0.0'
    r""" HRNet BackBone

   

    Args:
        model_input_shape (Tuple[int,int]) : default to (256,192)
        num_modules (List[int]) :number of HR fusion modules in each stage, Defaults to [1,1,4,3]
        num_blocks (List[int]) : number of BasicResModules for each branches of HR_fusion_modules,  Defaults to [4,4,4,4]
        branches_channels (List[int]) :  number of channels in each branches,  Defaults to [32,64,128,256]
            it's a ascending Order, i.e. [P2:32, P3:64, P4:128, P5:256]
        interpolation (str) :  Whether to use depthwise separable convolution in bottleneck blocks. Defaults to False.
        psa_type (str) :  Whether to use psa in basic ResModules,  default to None 
        multiscale_output (bool) :  Whether to  outputs multi-features, default to False 
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in DepthwiseConvModule and ConvModule, defaults to 'silu'.
        data_preprocessor (dict) = default to None
        depoly (bool): determine depolyment config for each cell . default to None, 
                depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfgs depend on above args.
                depoly = True => to use deployment config, only conv layer will be bulit
                depoly = False => to use training config() ,
    References:
            - [Based on implementation of 'HRNet' @mmPose] 
            (https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/backbones/hrnet.py)
            - [Paper : 'High-Resolution Representations for Labeling Pixels and Regions'] 
            (https://paperswithcode.com/paper/190807919)

    Note :
       - branches_channels = [32,64,128,256]@W32,  [48,96,192,384]@48
       - 
       - 
       - 
    Example:
        '''Python

        'HRNetW32'  
        model = HRNet(
                model_input_shape=(256,192),         
                num_modules = [1,1,4,3],
                num_blocks  = [4,4,4,4],
                branches_channels = [32,64,128,256],
                interpolation  = 'bilinear',
                psa_type = 'p',
                multiscale_output= False,
                data_preprocessor = None,
                activation = 'relu',
                bn_epsilon = 1e-5,
                bn_momentum = 0.9,
                name = "HRNetW32"
        )   
        model.summary(200)
    """
    def __init__(self,      
        model_input_shape : Tuple[int,int]=(256,192),
        num_modules : List[int] = [1,1,4,3],
        num_blocks : List[int] = [4,4,4,4],
        branches_channels : List[int] = [32,64,128,256],
        psa_type : Optional[str] = None,
        interpolation : str = 'bilinear',
        multiscale_output :bool = False,
        data_preprocessor: dict = None,
        name =  'HRNet',
        **kwargs):    

        self.multiscale_output = multiscale_output
        self.num_modules = num_modules
        self.branches_channels = branches_channels
        self.num_blocks = num_blocks
        self.num_stages = len(branches_channels)
        self.interpolation = interpolation
        self.psa_type = psa_type

        super().__init__(input_size =  (*model_input_shape,3),
                        data_preprocessor = data_preprocessor,
                        name = name, **kwargs)
         
    def make_stage(self,
                   xs : list, 
                   num_modules : int,
                   num_blocks : int,
                   branches_channels : List[int],
                   use_bottleneck_blocks : bool = False,
                   multiscale_output : bool=True,
                   name : str = 'stage'):
        
        assert (len(xs)==len(branches_channels)), \
        'len(xs) should be equal to len(branch_channels)'
      
        for i in range(num_modules):
            xs = HR_BaseBlock(
                    branches_channels = branches_channels,
                    num_blocks = num_blocks,
                    use_bottleneck_blocks = use_bottleneck_blocks,
                    multiscale_output = multiscale_output if i==num_modules-1 else True,
                    interpolation  = self.interpolation,
                    name  = name+f"_HRModule{i+1}", 
                    psa_type = self.psa_type, 
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy  = self.deploy,
            )(xs) 
        return xs
    
    def call(self,  
            x:tf.Tensor,
            training : Optional[bool]=None)->tf.Tensor:
        'stem'
        for i in range(2):
            x = Conv2D_BN(
                    filters = 64,
                    kernel_size=3,
                    strides=2,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy=self.deploy,
                    name = f'stem_ConvBn{i+1}'
            )(x)
  
        xs = [x]
        'STAGE1~4'
        for i in range(self.num_stages):
            stage_id = i+1
            multiscale_output = self.multiscale_output if i==self.num_stages-1 else True
       
            if stage_id == 1 :
                is_bottleneck = True
                branches_filters = [xs[0].shape[-1]*4]
            else:
                is_bottleneck = False 
                branches_filters = self.branches_channels[:(stage_id)]
                'Transition layer to expand features'
                xs = HR_TransitionBlock(
                    output_channels=branches_filters,
                    name  =f"Stage{stage_id}_TransModule",
                    deploy = self.deploy,  
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                )(xs) 
     
            xs = self.make_stage(
                xs,
                branches_channels=branches_filters,
                num_modules=self.num_modules[i],
                num_blocks=self.num_blocks[i],
                use_bottleneck_blocks = is_bottleneck,
                multiscale_output=multiscale_output,
                name=f'Stage{stage_id}'
            )  
        return xs if self.multiscale_output else xs[0]