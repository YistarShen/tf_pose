
import tensorflow as tf
#from lib.layers import GroupNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, Activation,LayerNormalization
from tensorflow_addons.layers import AdaptiveAveragePooling2D

class EfficientMultiScaleAttention(tf.keras.layers.Layer):
    VERSION = '1.0.0'

    R""" EfficientMultiScaleAttention(EMA)
    
    https://blog.csdn.net/DM_zx/article/details/132707712



    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    #print(out.shape)
    x = Input(shape=(64,48,256))
    out = EfficientMultiScaleAttention(
        groups  = 8, 
        ln_epsilon = 1e-3,     
        name='EMA'
    )(x)

    model = Model(x, out)
    model.summary(150)
    model.get_layer('EMA').get_config()
    model.save('test')
    jit_model = tf.function(model, jit_compile=True)
    tensor = tf.ones(shape=(1,64,48,256))
    %timeit _ = jit_model(tensor)
    """
    def __init__(
      self, 
      groups : int = 1, 
      ln_epsilon : float = 1e-3,
      **kwargs 
    ):
        super().__init__(**kwargs)
        self.groups = groups
        self.ln_epsilon = ln_epsilon

    def build(self, input_shape):

        _, self.h, self.w, self.c  = input_shape
        self.softmax = Activation('softmax')
        self.gap = GlobalAveragePooling2D(name='gap')
        
        self.pool_h = AdaptiveAveragePooling2D(output_size=(self.h,1))
        self.pool_w = AdaptiveAveragePooling2D(output_size=(1,self.w))

        if True:
            #https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
            #here, equivalent with LayerNorm
            self.gn = LayerNormalization(
                epsilon= self.ln_epsilon
            )
        else:
            self.gn = GroupNormalization(
                self.c//self.groups, epsilon= self.ln_epsilon
            )


        self.dims = (self.c//self.groups)
     
        self.conv1x1 =Conv2D(
            ( self.dims ),
            kernel_size=1,
            strides = 1,
            use_bias= True , 
            name="conv1x1"
        )
        self.conv3x3 =Conv2D(
            ( self.dims ),
            kernel_size=3,
            strides = 1,
            use_bias= True , 
            padding="same",
            name="conv3x3"
        )


    def call(self,x):
        r""" 
        
        
        """
        group_x = tf.reshape(
            x,[-1, self.h, self.w, self.groups,  self.dims ]
        )#(b,h,w,g,c//g)
        group_x = tf.transpose(
            group_x,[0,3,1,2,4]
        )#(b,h,w,g,c//g) => (b,g,h,w,c//g)
        group_x = tf.reshape(
            group_x,[-1, self.h, self.w,  self.dims ]
        )#(bg,h,w,c//g)

        x_h = self.pool_h(group_x) #(bg, h, 1, c//g)
        x_w = self.pool_w(group_x) #(bg, 1, w, c//g)
        x_w = tf.transpose(
            x_w, [0,2,1,3]
        )#(bg, w, 1, c//g)


        hw = tf.concat([x_h, x_w], axis=1) #(bg, h+w, 1, c//g)
        hw = self.conv1x1(hw) #(bg, h+w, 1, c//g)
        x_h, x_w = tf.split(hw, [self.h, self.w], axis=1) #(bg,h,1,c//g) and (bg,w,1,c//g)
        x_h = tf.nn.sigmoid(x_h)#(bg,h,1,c//g) 
        x_w = tf.nn.sigmoid(tf.transpose(x_w, [0,2,1,3])) #(bg,w,1,c//g)=> (bg,1,w,c//g)
        

        x1 = self.gn(
            group_x * x_h * x_w,
        ) #(bg,h,w,c//g)
        x2 = self.conv3x3(group_x) #(bg,h,w,c//g)

        x11 = self.softmax( self.gap(x1) ) #(bg,1,1,c//g)
        x11 = tf.reshape(
            x11, [-1,1, self.dims]
        ) #(bg,1,c//g)

        x12 = tf.reshape(
            x2, [-1,self.h*self.w,  self.dims]
        ) #(bg,h*w,c//g)
        x12 = tf.transpose(x12,[0,2,1]) #(bg,c//g,h*w)

        x21 = self.softmax( self.gap(x2) ) #(bg,1,1,c//g)
        x21 = tf.reshape(
            x21,[-1,1, self.dims ]
        ) #(bg,1,c//g)

        x22 = tf.reshape(x1, [-1,self.h*self.w,  self.dims ]) #(bg,h*w,c//g)
        x22 = tf.transpose(x22,[0,2,1]) #(bg,c//g,h*w)

        weights_1 = tf.matmul(x11, x12) # (bg,1,c//g)@(bg,c//g,h*w) => (bg,1, h*w)
        weights_2 = tf.matmul(x21, x22) # (bg,1,c//g)@(bg,c//g,h*w) => (bg,1, h*w)
        weights = tf.reshape(
            weights_1 + weights_2, [-1,self.h,self.w,1]
        )# (bg,1,h*w) => (bg,h,w,1)

        out = group_x * tf.nn.sigmoid(weights) #(bg,h,w,c//g)*(bg,h,w,1) =>  (bg,h,w,c//g)

        out = tf.reshape(
            out, [-1,self.groups,self.h, self.w,  self.dims]
        ) #(bg,h,w,c//g) => (b, g,h,w,c//g)
        out = tf.transpose(
            out, [0,2,3,1,4]
        ) #(b,h,w,g,c//g)
        out = tf.reshape(
            out, [-1,self.h, self.w, self.c]
        ) ##(b,h,w,g,c//g) => (b,h,w,c)
        return out 
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "groups"  : self.groups,
                "ln_epsilon": self.ln_epsilon,
            }
        )


        return config 