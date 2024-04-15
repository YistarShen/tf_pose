
from lib.Registers import MODELS
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, DepthwiseConv2D, BatchNormalization,LayerNormalization,Activation
from tensorflow.keras.layers import Concatenate, Reshape, UpSampling2D
from lib.models.modules import BaseModule, CSPLayerWithTwoConv
from lib.layers import Conv2D_BN
from lib.layers.convolutional import MaxPoolAndStrideConv2D


#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
@MODELS.register_module()
class PathAggregationFPN_C2f(BaseModule):
    VERSION = '1.0.0'
    r"""Implements path aggregation fpn (pafpn) for object detection used in YOLOv8

    # 9: p5 1024 ---+----------------------+-> 21: out2 1024
    #               v [up 1024 -> concat]  ^ [down 512 -> concat]
    # 6: p4 512 --> 12: p4p5 512 --------> 18: out1 512
    #               v [up 512 -> concat]   ^ [down 256 -> concat]
    # 4: p3 256 --> 15: p3p4p5 256 --------+--> 15: out0 128
    # features: [p3, p4, p5]

                csp_depth       P3P4P5_in_channels      P3P4P5_out_channels   P3P4P5_shape
    ------------------------------------------------------------------------------------------
    YOLOv8-x       3            [320, 640, 640]         [320, 640, 640]       [80, 40, 20]
    YOLOv8-l       3            [256, 512, 512]         [256, 512, 512]       [80, 40, 20]
    YOLOv8-m       2            [192, 384, 576]         [192, 384, 576]       [80, 40, 20]
    YOLOv8-s       1            [128, 256, 512]         [128, 256, 512]       [80, 40, 20]
    YOLOv8-n       1            [ 64, 128, 256]         [ 64, 128, 256]       [80, 40, 20]



    
    Args:
        csp_depth (int) : num bottleneck block used in c2f module. Defaults to 3.
        only_topdown (bool) : whether use bottom up network, if Ture, fpn only use topdown network. default to False.
        simple_downsample (bool) : whether to use MaxPoolAndStrideConv2D layer for downsample, default to False.
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in DepthwiseConvModule and ConvModule, defaults to 'swish'.
        depoly (bool): determine depolyment config for each cell . default to None, 
                depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfgs depend on above args.
                depoly = True => to use deployment config, only conv layer will be bulit
                depoly = False => to use training config() ,
    References:

            - [Based on implementation of 'apply_path_aggregation_fpn' @keras-cv ]
                     (https://github.com/keras-team/keras-cv/blob/832e2d9539fd29a8348f38432673d14c18981df2/keras_cv/models/object_detection/yolo_v8/yolo_v8_detector.py#L189)
            - [Inspired on implementation of 'YOLOv8PAFPN' @mmyolo] 
                    (https://github.com/open-mmlab/mmyolo/blob/8c4d9dc503dc8e327bec8147e8dc97124052f693/mmyolo/models/necks/yolov8_pafpn.py)                    
            - [Inspired on 'path_aggregation_fpn' @leondgarse] 
                    (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/yolov8/yolov8.py)

    Note :
       - output channels = input channels
       - CSPLayerWithTwoConv doesn't use shortcut
       - 
       - 
    Example:
        -------------------------------------------------------
        '''Python
        from lib.models.necks import PathAggregationFPN_C2f as PAFPN_C2f
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input

        #YOLOv8-n pafpn neck
        neck = PAFPN_C2f(
                csp_depth  = 1,
                only_topdown = False,
                simple_downsample  = True,
                deploy = False,
                bn_epsilon = 1e-5,
                bn_momentum = 0.9,
                activation = 'swish',
                name='neck')

        feat_p3 = Input((80,80,128))
        feat_p4 = Input((40,40,256))
        feat_p5 = Input((20,20,512))
        features = [feat_p3,feat_p4,feat_p5]
        out = neck(features)
        model = Model(features, out)
        model.summary(200, expand_nested=True)

        -------------------------------------------------------
        '''Python
        from lib.models import NECKS
        neck_cfg = dict(
                type = 'PathAggregationFPN_C2f',
                csp_depth  = 3,
                only_topdown = False,
                simple_downsample  = True,
                deploy = False,
                bn_epsilon = 1e-5,
                bn_momentum = 0.9,
                activation = 'swish',
                name='neck')
        NECKS.build(neck_cfg)

    """
    def __init__(self,
        csp_depth : int = 3,
        only_topdown : bool = False,
        simple_downsample : bool = True,
        activation : str='swish',
        name=None, 
        **kwargs):
        super().__init__(name=name, 
                        activation=activation, 
                        **kwargs)
        self.csp_depth = csp_depth
        'basic config'
        self.half_mode = only_topdown
        self.simple_downsample = simple_downsample
       
    def call(self, features):

        upsamples = [features[-1]]
        
        state =self.name+f"PAFPN_Up_p{len(features) + 2}"
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

            feat = CSPLayerWithTwoConv(
                    out_channels = feat.shape[-1],
                    expand_ratio= 0.5,
                    kernel_sizes = [3,3],
                    csp_depthes = self.csp_depth,
                    use_shortcut = False,
                    use_depthwise  = False,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy = self.deploy,
                    name = f'{state}_c2f'
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

            if self.simple_downsample:
                nn = Conv2D_BN(
                        filters = downsamples[-1].shape[-1],
                        kernel_size=3,
                        strides=2,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        activation = self.act_name,
                        deploy = self.deploy,
                        name=f'{state}_{feat_to_down}_conv'
                )(downsamples[-1])              

            else:
                nn = MaxPoolAndStrideConv2D(
                        in_channels_ratio = 1.0,
                        strides=2, 
                        use_bias =False,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        activation = self.act_name,
                        name=f'{state}_{feat_to_down}_MP'
                )(downsamples[-1]) 

            nn = Concatenate(
                axis=-1, name=f'{state}_concat'
            )([feat, nn]) 

            feat = CSPLayerWithTwoConv(
                    out_channels = feat.shape[-1],
                    expand_ratio= 0.5,
                    kernel_sizes = [3,3],
                    csp_depthes = self.csp_depth,
                    use_shortcut = False,
                    use_depthwise  = False,
                    bn_epsilon = self.bn_epsilon,
                    bn_momentum = self.bn_momentum,
                    activation = self.act_name,
                    deploy = self.deploy,
                    name = f'{state}_c2f'
            )(nn) 

            downsamples.append(feat)
        #downsamples(P3P4P5) : [  P3:(80,80,128), P4:(40,40,256), P5:(20,20, 256) ]
        return downsamples
    

#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------