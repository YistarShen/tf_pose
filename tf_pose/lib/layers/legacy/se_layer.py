
import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D,  DepthwiseConv2D, Conv2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, concatenate, Input, Activation
from tensorflow.keras.models import Model


############################################################################
#
#
############################################################################
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, channels, name=None, **kwargs):
        super(ChannelAttention, self).__init__(name=name)
        "ChannelAttention"
        self.channels = channels

    def build(self, input_shape):

        self.n, self.h, self.w, self.c = input_shape.as_list()

        self.gap = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.fc =  tf.keras.layers.Conv2D(self.c , (1,1), strides=(1,1), padding='same', use_bias=False)
        self.act = tf.keras.layers.Activation('hard_sigmoid')
        self.mul = tf.keras.layers.Multiply()

    def call(self, x):
        assert x.shape[-1]==self.channels," channels must be x.shape[-1] @ChannelAttention"
        att = self.gap(x)
        att = self.fc(att)
        att = self.act(att)
        out =  self.mul([x,att])
        return out

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update(
                {
                "channels": self.channels,
                "activation" : 'hard_sigmoid',
                }
        )
        return config  