'tf layers'
from typing import Optional
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import Concatenate, Reshape
import tensorflow as tf
#----------------------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------------------  
class DenseFuseLayer(tf.keras.layers.Layer):
    VERSION = '1.0.1'
    r""" DenseFuseLayer (DFL) used in bbox regression head of YOLOv8
    support switch_to_deploy

    Architecture :

     -------------------------------------------------------------------------------------

    References:
            - [Based on implementation of 'DFL' @ultralytics] 
              (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L15)
            - [inspired on implementation of 'decode_regression_to_boxes' @keras-cv] 
              (https://github.com/keras-team/keras-cv/blob/832e2d9539fd29a8348f38432673d14c18981df2/keras_cv/models/object_detection/yolo_v8/yolo_v8_detector.py#L288)
            - [inspired on method of non-trainable-convolution-filter] 
              (https://stackoverflow.com/questions/71754180/how-do-you-implement-a-custom-non-trainable-convolution-filter-in-tensorflow)
          
    Args:
        reg_max (int) : The maximum regression channels, default to 16
        use_conv (bool) : Whether to use conv2D, default to True
        name (str) : 'dfl'

    Note :
        - weights for dense or conv2d are non-trainable
        - initialized Conv2D kernel when reg_max=16: 
            tf.reshape( tf.range(start=0., limit=16, delta=1.), [1,1,16,1])
        - initialized Dense kernel when reg_max=16: 
            tf.reshape( tf.range(start=0., limit=16, delta=1.), [16,1])

    Examples:
    ```python

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    x = Input(shape=(8400,64))
    out = DenseFuseLayer(reg_max = 16,
                        use_conv = True,
            name = 'dfl')(x)
    model = Model(x, out)
    model.summary(100)  


    """
    def __init__(self,
            reg_max : int =16,
            use_conv : bool = True,   
            with_regression_channels : bool =False,       
            name : Optional[str] = None, 
            deploy : Optional[bool] = None,
            **kwargs):
        super().__init__(name=name, **kwargs)
        self.reg_max = reg_max
        self.use_conv = use_conv
        self.deploy = deploy
        self.with_regression_channels = with_regression_channels

    def build(self, input_shape):
        self.in_channels = input_shape[-1]

        if self.deploy :
            self.with_regression_channels = False #(deploy)

        proj = tf.range(start=0., limit=self.reg_max, delta=1.)
        if self.use_conv :
            kernel = tf.reshape(proj,[1,1,-1,1]).numpy()  
            # self.kernel = self.add_weight(name="proj",
            #             shape=(1,1, self.reg_max,1),
            #                 dtype=tf.dtypes.float32,
            #                 initializer=tf.constant_initializer(proj),
            #             trainable=False)
            'conv2d kernel : [filter_height, filter_width, in_channels//groups, out_channels]'
            self.dense = Conv2D(filters=1,
                              kernel_size=1, 
                              use_bias = False,
                              kernel_initializer=tf.constant_initializer(kernel), 
            )
            
        else:
            'dense kernel : [in_channels, out_channels]'
            kernel = tf.reshape(proj,[-1,1]).numpy()  
            self.dense = Dense(units=1,
                                use_bias=False,
                                kernel_initializer=tf.constant_initializer(kernel)
            )
            
        self.dense.trainable = False
        #self.reg = tf.keras.layers.Layer(name="reg_out")

        self.out_channels =  self.in_channels//self.reg_max + (self.in_channels if self.with_regression_channels else 0)
  
    
    def call(self, x):
        '''
        x : (b,h*w, in_channels=64)
        '''
        if self.with_regression_channels:
            reg_x  = x

        x = Reshape(
                [-1, self.in_channels//self.reg_max, self.reg_max],
                name='reshape_in'
        )(x)

        #x : (b,h*w, 64//16, 16) @reg_max=16
        x = tf.nn.softmax(
                x, axis=-1, name='softmax'
        )
        #x : (b,h*w, 64//16, 16) @reg_max=16

        # x = tf.nn.conv2d(x, 
        #                 self.kernel , 
        #                 strides=[1,1,1,1], 
        #                 padding='SAME', 
        #                 name='conv'
        #     )

        fused_x = self.dense(x)
        #x : (b,h*w, 64//16, 1) @reg_max=16

        fused_x = Reshape([-1, self.in_channels//self.reg_max],
                       name='reshape_fuse_out'
                )(fused_x) #x : (b,h*w, 64//16) @reg_max=16
        if self.with_regression_channels:
            fused_x = Concatenate( name="fuse_out_with_reg", axis=-1)([fused_x,reg_x]) 

        return tf.keras.layers.Layer(name='fuse_out')(fused_x)

    def switch_to_deploy(self):
        '''switch_to_deploy

        '''
        if self.deploy or self.deploy==None:
            return
        
        #'init a new dfl'
        self.built = False
        self.deploy = True
        'build new conv by seting input shape'
        super().__call__(tf.random.uniform(shape=(1,8400,self.in_channels)))
   

    def get_config(self):
        config = super().get_config()
        config.update(
                {
                    "reg_max": self.reg_max,
                    "with_regression_channels" : self.with_regression_channels,
                    "in_channels": self.in_channels,
                    "out_channels": self.out_channels,
                    "use_conv": self.use_conv
                }
        )
        return config