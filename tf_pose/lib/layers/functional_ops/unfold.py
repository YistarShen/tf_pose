'tf layers'
from tensorflow.keras.layers import  Conv2D
import tensorflow as tf


def __unfold_filters_initializer__(
            weight_shape , 
            dtype="float32"):
    #TensorFlow Conv2d kernel : [filter_height, filter_width, in_channels//groups, out_channels]
    kernel_size = weight_shape[0]
    kernel_out = kernel_size * kernel_size
    ww = tf.reshape( 
              tf.eye(kernel_out, dtype="float32"), 
              [kernel_size, kernel_size, 1, kernel_out]
    )
    #ww = np.reshape(np.eye(kernel_out, dtype="float32"), [kernel_size, kernel_size, 1, kernel_out])
    return ww
    
    
class UnFoldLayer(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    R"""  ExtractPatches(unfold)

    equivalent to nn.UnFold(x)

    torch_tensor = torch.from_numpy(np_arr).view(1,1,10,10)
    ds = nn.Unfold(kernel_size=3,stride=2, dilation=1, padding=1)
    torch_out = ds(torch_tensor).view(1,-1, 5, 5)

    tf_tensor = tf.reshape(np_arr,[1,10,10,1])
    pad_value_list = [[0, 0], [1, 1], [1, 1], [0, 0]]
    tf_tensor = tf.pad(tf_tensor, pad_value_list)
    tf_out = tf.image.extract_patches(
        tf_tensor,
        sizes=[1,3,3,1],
        strides=[1,2,2,1],
        rates=[1,1,1,1],
        padding='VALID'
    )

    
    References:
        - [Based on  implementation of  'CompatibleExtractPatches' @leondgarse]
          (https://github.com/leondgarse/keras_cv_attention_models/blob/6743a329cfcbf14b01320f2e86ad10a6c649080c/keras_cv_attention_models/common_layers.py#L575)

    
    Args:
        kernel_size (int) : the size of the sliding blocks, defaults to 3.
        strides (int) :  the stride of the sliding blocks in the input spatial dimensions, defaults to 1.
        dilation_rate (int) :  a parameter that controls the stride of elements within the neighborhood, defaults to 1. 
        use_conv (bool) : whether to use Conv2D to  do unfold ops, defaults to False 
        name (str) :'UnFold'

    Note:
        - PyTorch Conv2d kernel : [out_channels, in_channels//groups, filter_height, filter_width]
        - TensorFlow Conv2d kernel : [filter_height, filter_width, in_channels//groups, out_channels]

    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    x = Input(shape=(256,256,32))
    out = ExtractPatches(
            kernel_size = 3, 
            strides  = 1,
            dilation_rate  = 1,
            use_conv = False,
            name='UnFold'
    )(x)
    model = Model(x, out)
    model.summary(150)   

    """
    def __init__(self, 
                kernel_size : int = 3, 
                strides : int = 1,
                dilation_rate :  int = 1,
                use_conv : bool = False,
                compressed_output : bool = True,
                padding : str = "same",
                **kwargs ):
        super().__init__(**kwargs)

        self.k = kernel_size
        self.strides = strides
        self.rate = dilation_rate
        self.use_conv = use_conv
        self.padding = padding
        self.compressed = compressed_output

    def build(self, input_shape):
        _, self.H, self.W, self.c  = input_shape

        if self.strides>1 :
            self.hh, self.ww  =  (self.H//self.strides), (self.W//self.strides)
        else:
            self.hh, self.ww =  self.H, self.W
        
        if self.padding.lower() == "same":
            pad_value = self.k//2
            self.padding_list = [[0, 0], [pad_value, pad_value], [pad_value, pad_value], [0, 0]]
            self.H += pad_value * 2
            self.W += pad_value * 2
    
        if  self.use_conv  :
            #self.filters = self.k * self.k
            self.unfold_conv = Conv2D(
                    filters=pow(self.k, 2),
                    kernel_size=self.k,
                    strides=self.strides,
                    dilation_rate=self.rate,
                    padding="valid",
                    use_bias=False,
                    trainable=False,
                    kernel_initializer=__unfold_filters_initializer__,
                    name="unfold_conv",
            )
        #super().build(input_shape)

    def _Unfold(self, x):
        # x : (b, h,w, c)
        return tf.image.extract_patches(
                images=x,
                sizes=[1, self.k, self.k, 1],
                strides=[1, self.strides, self.strides, 1],
                rates=[1, self.rate, self.rate, 1],
                padding='VALID'
        ) # (b,h,w, c*k**2)
        
          

    def call(self,x):

        if hasattr(self, 'padding_list'):
            x = tf.pad(x, self.padding_list)

        if hasattr(self, 'unfold_conv'):
            merged_x  = tf.transpose(x, [0, 3, 1, 2]) # (b,c,H,W)
            merged_x = tf.reshape(merged_x, [-1, self.H, self.W, 1])  # (b,c,H,W) =>  # (b*c,H,W,1)
            conv_rr = self.unfold_conv(merged_x)  # (b*c,H,W,1) =>  (b*c,h,w, k**2)
            # out = tf.reshape(
            #     conv_rr, [-1, self.c, conv_rr.shape[1] * conv_rr.shape[2], self.filters]
            # ) #(b*c, h,w, k**2) => (b, c, h*w, k**2)
            #print(conv_rr.shape, self.hh, self.ww)
            out = tf.reshape(
                conv_rr, [-1, self.c, self.hh*self.ww,  pow(self.k,2)]
            ) #(b*c, h,w, k**2) => (b, c, h*w, k**2)
            out = tf.transpose(out, [0, 2, 3, 1])  # (b, h*w, k**2, c)

            # out = tf.reshape(
            #     out, [-1, conv_rr.shape[1], conv_rr.shape[2], pow(self.k,2)*self.c]
            # ) #  (b, h*w, k**2, c) =>  (b, h, w, k**2 *c)
            out = tf.reshape(
                out, [-1, self.hh, self.ww,  pow(self.k,2) * self.c]
            ) #  (b, h*w, k**2, c) =>  (b, h, w, k**2 *c)
        else:
           out = self._Unfold(x)  #(b, h, w, k**2 *c)

        if not self.compressed:
            out = tf.reshape(
                out, [-1, self.hh, self.ww, self.k, self.k, self.c]
            ) #  (b, h*w, k**2, c) =>  (b, h, w, k**2 *c)  

        return out

    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {
                "kernel_size": self.k,
                "strides": self.strides,
                "dilation_rate": self.rate,
                "use_conv": self.use_conv,
                "compressed_output": self.compressed,
                "non_compressed_output": (None, self.hh, self.ww, self.k, self.k, self.c),
                "padding" : self.padding
            }
        )
        return base_config