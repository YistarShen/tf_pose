from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from collections import OrderedDict
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Add, Concatenate, Resizing
from lib.Registers import MODELS
from lib.layers import Conv2D_BN, ADown
from lib.models.model_utils import make_divisible, make_round
from lib.models.modules import BaseModule
from lib.models.heads import YOLO_AnchorFreeHead
from lib.models.modules import RepNCSPELAN4
#------------------------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------------------------- 
class CBLinear(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r""" CBLinear

    """
    def __init__(
            self, 
            size_splits : list,
            kernel_size : int=1,
            strides: int =1,
            groups: int =1,
            use_bias:bool= False,
            **kwargs
    ):
        super().__init__(**kwargs)
        if isinstance(kernel_size, (list, tuple)) :
            self.kernel_size = kernel_size
        else :
            self.kernel_size = (kernel_size, kernel_size)

        if not isinstance(size_splits, list) :
            raise TypeError(
                "size_splits must be list type"
                f", but got {type(size_splits)} @ {self.__class__.__name__}"
            ) 
          
        self.size_splits = size_splits
        self.filters = sum(size_splits)
        self.strides = strides
        self.groups = groups
        self.use_bias = use_bias

    def build(self, input_shapes):

        if self.filters < 0: 
            _,_,_, in_channels = input_shapes
            self.filters = in_channels

        if self.size_splits == []:
            self.size_splits = [self.filters] 

        self.conv = Conv2D_BN(
                filters = self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                groups = self.groups ,
                use_bias = True,
                use_bn = False,
                activation = None,
                name = 'Conv'
        )

    def call(self, x):
        feat = self.conv(x)
        outs = tf.split(feat, num_or_size_splits= self.size_splits, axis=-1)
        # return list [Tensor1, Tensor2,....] or [Tensor] 
        return outs
    
#------------------------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------------------------- 

class CBFuse(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r""" CBFuse used in YOLOV9 AUX BRANCH

    """
    def __init__(
            self, 
            target_size :Tuple[int]  ,
            interpolation : str = 'nearest',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.interpolation = interpolation
        self.target_size = target_size
        

    def build(self, input_shapes):
        # input_shapes must be list type and channel last type
        # if self.target_size == -1 :
        #     self.target_size = input_shapes[0][1:3]
        print("test : ",input_shapes[1:])
        if not all([ size[-1]==input_shapes[0][-1] for size in input_shapes[1:]]):
            raise ValueError(
                "all input tensors must have same channels(last dim), "
                f"but got {input_shapes} "
            )

        self.resize_layers = [
            Resizing(
                    *self.target_size,
                    interpolation=self.interpolation ,
                    name=f'Resizing{i+1}'
            )for i in range(len(input_shapes)) 
        ]
        self.add_sum = Add(name = 'Add')   
        
    def call(self, xs):
        res = [resize(x) for x, resize in zip(xs, self.resize_layers) ]
        outs = self.add_sum(res)
        return outs
    

#------------------------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------------------------- 
@MODELS.register_module()
class  YOLOv9_AuxBranch(BaseModule):
    
    VERSION = '1.0.0'
    r""" GELAN (Generalized Efficient Layer Aggregation Network) used in YOLOv9
    GELAN = CSPNet + ELAN
    YOLOv9 = GELAN + Auxiliary Reversible Branch 

    GELAN : Main Branch
    Auxiliary Reversible Branch  : used by Programmable Gradient Information(PGI) technique 

    """
    
    #[ in_channels, out_channels, hidden_channels, num_blocks, add_identity, use_ADown]
    arch_settings = OrderedDict(
        P2 = [64, 256, 128, 1, True, False],
        P3 = [256, 512, 256, 1, True, True],
        P4 = [512, 512, 512, 1, True, True], 
        P5 = [512, 512, 512, 1, True, True]  
    )


    def __init__(self,  
        reg_max: int = 16, 
        num_classes : int=80,
        with_auxiliary_regression_channels : bool = False,            
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        expand_ratio: float = 0.5,
        use_depthwise: bool = False,
        show_multi_level_aux_info : bool = True,
        name =  'AuxBranch',
        *args, **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)

 
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.expand_ratio = expand_ratio
        self.use_depthwise = use_depthwise
        self.reg_max = reg_max
        self.num_classes = num_classes
        self.show_multi_level_aux_info = show_multi_level_aux_info
        self.with_auxiliary_regression = with_auxiliary_regression_channels

    def test(self, multi_level_aux:List[tf.Tensor]):
        print( 'multi_level_aux_info : \n')
        for level_aux in multi_level_aux:
            print(level_aux,"\n")

    # def _downsample(self, use_ADown: bool=True):
    #     if not use_ADown :
    #         x = Conv2D_BN(
    #             filters = hidden_channels,
    #             kernel_size=3,
    #             strides=2,
    #             bn_epsilon = self.bn_epsilon,
    #             bn_momentum = self.bn_momentum,
    #             activation = self.act_name,
    #             deploy  = None,
    #             name = self.name+ f'{p_feat}_DsConv'
    #         )(x)
    #     else:
    #             x = ADown(
    #             out_channels =  hidden_channels,
    #             bn_epsilon = self.bn_epsilon,
    #             bn_momentum = self.bn_momentum,
    #             activation = self.act_name,
    #                     name= self.name+  f'{p_feat}_DsConv'
    #             )(x) 

    def call(
            self, xs : List[tf.Tensor]
    )->List[tf.Tensor]:
        
        x = xs[0]
        xs_p = xs[1:]
        num_feat_to_fuse = len(xs_p)

        multi_level_aux_info = []
        size_splits = []
        for arch_setting_p, x_p in zip( list( self.arch_settings.items() )[-num_feat_to_fuse:] , xs_p):
            key , vals = arch_setting_p
            in_channels, out_channels, hidden_channels, num_bottleneck_block, add_identity, use_ADown = vals
            size_splits.append(make_divisible(hidden_channels, self.widen_factor))
            multi_level_aux_info.append(
                CBLinear(size_splits=size_splits, name= self.name+f'{key}_CBLinear')(x_p)
            )
        if self.show_multi_level_aux_info:
            print( 'multi_level_aux_info : \n')
            for level_aux in multi_level_aux_info:
                print(level_aux,"\n")


        xs = []
        'stem'
        stem_channels = make_divisible(self.arch_settings['P2'][0], self.widen_factor) 
        x = Conv2D_BN(
            filters = stem_channels,
            kernel_size=3,
            strides=2,
            bn_epsilon = self.bn_epsilon,
            bn_momentum = self.bn_momentum,
            activation = self.act_name,
            deploy  = None,
            name = self.name+'Stem_P1_Conv1'
        )(x)


        for i, (key, vals) in enumerate(self.arch_settings.items()):  
            to_fuse_feats = True  if i>=(len(list(self.arch_settings.keys()))-num_feat_to_fuse) else False
           
            in_channels, out_channels, hidden_channels, num_bottleneck_block, add_identity, use_ADown = vals
            out_channels = make_divisible(out_channels, self.widen_factor)
            hidden_channels = make_divisible(hidden_channels, self.widen_factor)
            num_bottleneck_block = make_round(num_bottleneck_block, self.deepen_factor)
            p_feat = f'Stage{key}'

            if to_fuse_feats:
                feat_idx = i- (len(list(self.arch_settings.keys()))-num_feat_to_fuse)
                feats_to_fuse = tf.nest.flatten([levels[feat_idx:feat_idx+1]for levels in multi_level_aux_info])
                x = ADown(
                        out_channels =  feats_to_fuse[0].shape[-1],
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        activation = self.act_name,
                        name= self.name+  f'{p_feat}_DsConv'
                )(x) 

                feats_to_fuse += [x] 

                x = CBFuse(
                   target_size=x.shape[1:3],name= self.name+f'{p_feat}_CBFuse'
                )(feats_to_fuse)

            else:
                x = Conv2D_BN(
                        filters = hidden_channels,
                        kernel_size=3,
                        strides=2,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        activation = self.act_name,
                        deploy  = None,
                        name = self.name+ f'{p_feat}_DsConv'
                )(x)


      


            # #downsample_channels =  if to_fuse_feats
            # if not use_ADown :
            #     x = Conv2D_BN(
            #             filters = hidden_channels,
            #             kernel_size=3,
            #             strides=2,
            #             bn_epsilon = self.bn_epsilon,
            #             bn_momentum = self.bn_momentum,
            #             activation = self.act_name,
            #             deploy  = None,
            #             name = self.name+ f'{p_feat}_DsConv'
            #     )(x)
            # else:
            #     x = ADown(
            #             out_channels =  hidden_channels,
            #             bn_epsilon = self.bn_epsilon,
            #             bn_momentum = self.bn_momentum,
            #             activation = self.act_name,
            #             name= self.name+  f'{p_feat}_DsConv'
            #     )(x) 


            # if to_fuse_feats:
            #     feat_idx = i- (len(list(self.arch_settings.keys()))-num_feat_to_fuse)
            #     feats_to_fuse = [x] + tf.nest.flatten([levels[feat_idx:feat_idx+1]for levels in multi_level_aux_info])
            #     x = CBFuse(
            #        target_size=x.shape[1:3],name= self.name+f'{p_feat}_CBFuse'
            #     )(feats_to_fuse)

             
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
                    name  = self.name+ f'{p_feat}_RepCSPELAN4',  
            )(x) 

            if to_fuse_feats:
                xs.append(x)
                
        return YOLO_AnchorFreeHead(
            reg_max= self.reg_max, 
            num_classes=self.num_classes,
            with_auxiliary_regression_channels = self.with_auxiliary_regression,
            bn_epsilon = self.bn_epsilon,
            bn_momentum = self.bn_momentum,
            activation = self.act_name,
            bbox_conv_groups = 4,
            name= self.name+ 'Head'
        )(xs)

#------------------------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------------------------- 


#------------------------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------------------------- 