from lib.Registers import MODELS
from lib.models.modules import BaseModule
from lib.layers import Conv2D_BN, RepVGGConv2D
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Activation, Concatenate, Reshape
#----------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------
@MODELS.register_module()
class YOLO_AnchorBaseHead(BaseModule):
    VERSION = '1.0.0'
    r""" YOLO_AnchorBaseHead
                           
                P3P4P5_in_channels     reparam_conv
    ---------------------------------------------------
    YOLOv7-tiny [ 64,128,256]          False             
    YOLOv7      [128,256,512]          True     
    YOLOv7-x    [160,320,640]          False 


    
    Args:
        num_anchors (int) : Number of anchors for each anchor point; 
            it depends on your encoder , default to 3  
        num_classes (int) : Number of categories excluding the background
            category. default to 80.
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in DepthwiseConvModule and ConvModule, defaults to 'silu'.
        depoly (bool): determine depolyment config for each cell . default to None, 
                depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfgs depend on above args.
                depoly = True => to use deployment config, only conv layer will be bulit
                depoly = False => to use training config() 


    Note :
       -  
       - 
       - 
       - 
    References:

            - [Based on implementation of 'yolov7_head' @leondgarse ]
                     (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/yolov7/yolov7.py#L279)

    Example:
        -------------------------------------------------------
        '''Python
        from lib.models.heads import YOLO_AnchorBaseHead
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        head = YOLO_AnchorBaseHead(
                            num_anchors= 3, 
                            num_classes=80,
                            use_reparam_conv_head = True,
                            output_dict_format = False,
                            activation = 'swish',
                            bn_epsilon = 1e-5,
                            bn_momentum = 0.9,
                            deploy =  False,
                            scope_marks = '_',
                            wrapped_model = False,
                            name='YOLOv7_Head'
        )
                
        feat_p3 = Input((80,80,64))
        feat_p4 = Input((40,40,128))
        feat_p5 = Input((20,20,256))
        features = [feat_p3,feat_p4,feat_p5]
        out = head(features)
        model = Model(features, out)
        model.summary(200, expand_nested=True)

        -------------------------------------------------------
        '''Python
        from lib.models import HEADS
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        head_cfg = dict(
                    type = 'YOLO_AnchorBaseHead',
                    num_anchors= 3, 
                    num_classes=80,
                    use_reparam_conv_head = True,
                    dict_format_output = True,
                    activation = 'swish',
                    bn_epsilon = 1e-5,
                    bn_momentum = 0.9,
                    deploy =  False,
                    scope_marks = '_',
                    wrapped_model = True,
                    name='YOLOv7_Head'
        )

        head = HEADS.build(head_cfg)
        feat_p3 = Input((80,80,64))
        feat_p4 = Input((40,40,128))
        feat_p5 = Input((20,20,256))
        features = [feat_p3,feat_p4,feat_p5]
        out = head(features)
        model = Model(features, out)
        model.summary(200, expand_nested=True)
    """         

          
    def __init__(self,
            num_anchors : int=3,  
            num_classes : int=80,
            use_reparam_conv_head = True,
            activation : str='silu',
            dict_format_output : bool = False,
            dense_type_output : bool = False,
            name=None, 
            **kwargs):
        super().__init__(name=name, 
                        activation=activation, 
                        **kwargs)
        #self.box_regression_channels = box_regression_channels

        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.reparam_conv_head = use_reparam_conv_head
        self.dict_format = dict_format_output
        self.dense_type = dense_type_output

    def build(self, input_shape):
        ''' 
        input_shape : [P3:(),P4:()]
        '''
        self.anchors = []
        for shape in (input_shape):
            self.anchors.append( shape[1]*shape[2]*self.num_anchors)


    def call(self, features):
        ''' 
        features : [P3 :(b, 80,80,128), P4 :(b, 40,40,256), P5 :(b, 20,20,512)]
        '''

        outputs = {} if self.dict_format else []

        for idx, feature in enumerate(features):
            cur_name = self.name + f"P{idx+3}"
            #--------first conv--------------------
            if self.reparam_conv_head :
                feature = RepVGGConv2D(
                        filters = feature.shape[-1]*2,
                        kernel_size=3,
                        use_bias = False,
                        use_depthwise =False,
                        groups=1,
                        activation = self.act_name,
                        deploy = self.deploy,
                        name = f'{cur_name}_RepVGGConv'
                )(feature) 
            else:
                feature = Conv2D_BN(
                        filters = feature.shape[-1]*2,
                        kernel_size=3,
                        bn_epsilon = self.bn_epsilon,
                        bn_momentum = self.bn_momentum,
                        activation = self.act_name,
                        deploy= self.deploy,
                        name=f'{cur_name}_conv'
                )(feature) 
            #--------2nd conv--------------------        
            feature = Conv2D(
                        filters=self.num_anchors*(4+self.num_classes+1), 
                        kernel_size=(1,1), 
                        strides=1,
                        padding="same",
                        name=f'{cur_name}_pwconv', 
                        dtype=tf.float32
            )(feature)

            #(b,h,w,3*(4+num_classes+1) )
            feature = Reshape(
                    target_shape=[*feature.shape[1:3], self.num_anchors,-1], 
                    name=f'{cur_name}_Reshape', 
                    dtype=tf.float32
            )(feature)


            feature = Activation(
                'linear', name=f"P{idx+3}_output"
            )(feature)

            if self.dict_format :
                outputs[f'{idx+3}_output'] = feature
            else:    
                outputs.append(feature)

        return outputs  