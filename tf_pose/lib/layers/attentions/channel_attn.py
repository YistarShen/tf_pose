import tensorflow as tf
from tensorflow_addons.layers import AdaptiveAveragePooling2D
from tensorflow.keras.layers import Multiply, Conv2D,  GlobalAveragePooling2D, Activation

class ChannelAttention(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r"""ChannelAttention

    basic Channel Attention  without squeeze and excition

    References:
            - [Channel-attention module ] (https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet)
            - [ChannelAttention(nn.Module) @ultralytics ] (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py)
    Args:
        use_bias (bool) : Whether use conv's bias, default to True,
        use_adpt_pooling (bool) : Whether use AdaptiveAveragePooling2D from tensorflow_addons , 
                                  if not, module will use GlobalAveragePooling2D(keepdims=True)
        activation (str) :  activation function name that only support available class in tf.keras.activations, default to 'sigmoid'

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
                use_bias : bool = True,
                use_adpt_pooling : bool = False,
                activation = 'sigmoid',
                name: str='ChannelAttention',
                **kwargs):
        super(ChannelAttention, self).__init__(name=name)
        self.use_bias = use_bias
        self.adpt_pooling = use_adpt_pooling
        self.activation = activation

    def build(self, input_shape):
        
        _,_,_,self.channels= input_shape
        if self.adpt_pooling :
            self.gap = AdaptiveAveragePooling2D(output_size=1,  name='aap')
        else:
            self.gap = GlobalAveragePooling2D(keepdims=True, name='gap')
       
        self.fc = Conv2D(self.channels, 1, 1, use_bias=self.use_bias)
        self.act = Activation(self.activation, name=f'{self.activation}')
        self.mul = Multiply(name='mul')

    def call(self, inputs):
        attn = self.gap(inputs)
        attn = self.fc(attn)
        attn = self.act(attn)
        return self.mul([inputs,attn])

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update(
                {
                "use_bias": self.use_bias,
                "adpt_pooling": self.adpt_pooling,
                "activation" : self.activation
                }
        )
        return config 