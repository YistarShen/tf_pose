
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Concatenate, Multiply, Activation

class SpatialAttention(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r"""SpatialAttention

     x*self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

    References:
            - [SpatialAttention(nn.Module) @ultralytics ] (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py)

    Args:
        use_bias (bool) : default to True,
        kernel_size (int) : Conv2D(1, kernel_size, 1) to get  Spatial features , default to 7

    Examples:
    ```python

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    x = Input(shape=(256,256,128))
    out = ChannelAttention( use_bias = True,
                            use_adpt_pooling = True,
                            name='ChannelAttention')(x)
    model = Model(x, out)
    model.summary(150)

    '''end

    """
    def __init__(self, 
                kernel_size : int = 7,
                use_bias : bool = True,
                activation : str = 'sigmoid',
                name: str='SpatialAttention',
                **kwargs):
        super(SpatialAttention, self).__init__(name=name)

        assert kernel_size in (3, 7), f'kernel size must be 3 or 7 @{self.__class__.__name__}'
        self.kernel_size = (kernel_size, kernel_size)
        self.use_bias = use_bias
        self.activation = activation

    def build(self, input_shape):
        
        padding = (self.kernel_size[0]//2, self.kernel_size[1]//2)
        if max(padding)>0:
            self.zero_padding2D = ZeroPadding2D(padding=padding,name='pad')

        self.conv = Conv2D(filters = 1,
                        kernel_size=self.kernel_size,
                        strides=1,
                        use_bias=self.use_bias,
                        padding="valid",
                        name="conv")
        
        self.concat = Concatenate(axis=-1, name='concat')
        self.act = Activation(self.activation, name=f'{self.activation}')
        self.mul = Multiply(name='mul')

    def call(self, inputs):
        print(inputs)
        ''' 
        inputs : (b,h,w,c) 
        return : (b,h,w,c) 
        '''
        x_mean = tf.math.reduce_mean(inputs,axis=-1,keepdims=True) # (b,h,w,c) => (b,h,w,1)
        x_max = tf.math.reduce_max(inputs,axis=-1,keepdims=True)  # (b,h,w,c) => (b,h,w,1)
     
        x = self.concat([x_mean,x_max])  #concat( [(b,h,w,1) ,(b,h,w,1)] => (b,h,w,2) 
        if hasattr(self,'zero_padding2D'):
            x = self.zero_padding2D(x)

        attn = self.conv(x) #(b,h,w,2) => (b,h,w,1) 
        attn = self.act(attn)
        
        return self.mul([inputs,attn])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update(
                {
                "use_bias": self.use_bias,
                "zero_padding2D" : self.zero_padding2D,
                "kernel_size": self.kernel_size,
                "activation": self.activation,
                }
        )
        return config 
    