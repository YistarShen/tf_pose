
'tf layers'
from typing import Optional, Tuple
from tensorflow.keras.layers import Conv2DTranspose
import tensorflow as tf
from .unfold import __unfold_filters_initializer__


class FoldLayer(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    R"""  FoldLayer(fold)
    
    equivalent to nn.Fold()(tensor)

    np_arr = np.arange(1, 101, 1, dtype=float)
    '--------torch-------------------------------'
    torch_tensor =torch.from_numpy(np_arr).view(1,1,10,10)
    torch_unfold_out = nn.Unfold(
        kernel_size=3,stride=2, dilation=1, padding=1
    )(torch_tensor)   #(1,9,5*5)
    torch_fold_out = nn.Fold(
        output_size=(10,10),stride=2, kernel_size=3, dilation=1, padding=1
    )(torch_unfold_out)  #(1,1,10,10)

    
    '--------tf-------------------------------'
    tf_tensor = tf.reshape(np_arr,[1,10,10,1])     
    tf_tensor = tf.pad(
        tf_tensor,  [[0, 0], [1, 1], [1, 1], [0, 0]]
    )
    tf_unfold_out = tf.image.extract_patches(
        tf_tensor,
        sizes=[1,3,3,1],
        strides=[1,2,2,1],
        rates=[1,1,1,1],
        padding='VALID'
    )
    tf_fold_out = FoldLayer(
        output_size=(10,10),stride=2, kernel_size=3, dilation=1
    )(tf_unfold_out)


    
    References:
        - [Based on  implementation of  'fold_by_conv2d_transpose' @leondgarse]
          (https://github.com/leondgarse/keras_cv_attention_models/blob/6743a329cfcbf14b01320f2e86ad10a6c649080c/keras_cv_attention_models/common_layers.py#L575)

    
    Args:
        kernel_size (int) : the size of the sliding blocks, defaults to 3.
        strides (int) :  the stride of the sliding blocks in the input spatial dimensions, defaults to 1.
        dilation_rate (int) :  a parameter that controls the stride of elements within the neighborhood, defaults to 1. 
        name (str) :'Fold'

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
                output_shape : Optional[Tuple[int]],
                kernel_size : int = 3, 
                strides : int = 2,
                dilation_rate :  int = 1,
                compressed_input = True,
                padding : str = "same",
                **kwargs ):
        
        super().__init__(**kwargs)


        self.target_size = output_shape
        self.k = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.compressed = compressed_input
        self.padding = padding


    def build(self, input_shape):


        if len(input_shape) != 4 and  len(input_shape) != 6: 
            raise ValueError(
               "'FoldLayer' only support input.shape.rank=4 or 6, i.e. (b,h,w,c) or (b,h,w,k,k,cc)" 
            )
        
        if self.compressed :
            _, self.h, self.w, dims  = input_shape
            self.channels = dims//pow(self.k,2) 
        else  : 
            _, self.h, self.w, _, _, self.channels  = input_shape

        if self.target_size is None:
            self.target_size = [self.h*self.strides, self.w*self.strides]

        self.pad_value = self.k//2 if self.padding.lower() == "same" else 0
        self.H = self.target_size[0] +self.pad_value*2
        self.W = self.target_size[1] +self.pad_value*2


        self.fold_Conv2DTranspose= Conv2DTranspose(
            filters=1,
            kernel_size=self.k,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            padding="valid",
            output_padding=self.pad_value,
            use_bias=False,
            trainable=False,
            kernel_initializer=__unfold_filters_initializer__,
            name="fold_convtrans",
        )

    def call(self,patches):

        conv_rr = tf.reshape(
            patches, [-1, self.h*self.w, pow(self.k,2), self.channels]
        )#(b,h*w, k**2, channels)
        conv_rr = tf.transpose(
            conv_rr, [0, 3, 1, 2]
        )  # [batch, channnels, h* w, k**2]
        conv_rr = tf.reshape(
            conv_rr, [-1, self.h, self.w, pow(self.k,2)]
        )# [batch*channnels, h, w, k**2]

        convtrans_rr = self.fold_Conv2DTranspose(conv_rr) #[batch*channnels, H, W, 1]

        out = tf.reshape( 
            convtrans_rr[..., 0], [-1, self.channels, self.H, self.W]
        )#[batch, channnels, H, W]
        out = tf.transpose(out, [0, 2, 3, 1]) #[batch, H, W, channnels]
        out = tf.slice(
            out,
            begin = [0,self.pad_value, self.pad_value,0],
            size =  [-1,self.target_size[0],self.target_size[1],-1]
        )
        # out = out[:, self.pad_value : self.target_shape[0], self.pad_value : self.target_shape[1], : ]  #[batch, H, W, channnels]
        return out  

    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {
                "kernel_size": self.k,
                "strides": self.strides,
                "dilation_rate": self.dilation_rate,
                "output_shape" : (*self.target_size, self.channels),
                "padding" : self.padding,
                "compressed_input" : self.compressed
              
            }
        )
        return base_config
