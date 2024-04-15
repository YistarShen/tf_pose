'tf layers'
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tensorflow.keras.layers import Conv2DTranspose, Dense, Conv1D, Conv2D, DepthwiseConv2D, BatchNormalization,LayerNormalization,Activation
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, ZeroPadding1D, ZeroPadding2D
from tensorflow.keras.layers import Concatenate, Multiply, Add, Reshape, Resizing
from tensorflow import Tensor
import math
import tensorflow as tf

#---------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------
class Conv2D_BN(tf.keras.layers.Layer):
    VERSION = '2.0.1'
    r""" Conv2D_Layer (conv2d_no_bias)
    support fuse conv+bn as one conv layer by switch_to_deploy

    Return : 
        tf.Tensor

    References:
        - [Based on @hoangthang1607's  implementation of repvgg] (https://github.com/hoangthang1607/RepVGG-Tensorflow-2/blob/main/repvgg.py)
        - [Inspired by  @model_surgery's model_surgery] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/model_surgery/model_surgery.py)
        - [Inspired by  @notplus's MobileOne-TF2] (https://zhuanlan.zhihu.com/p/543101751) (https://github.com/notplus/MobileOne-TF2/blob/main/mobileone.py)
    
    Args:
        filters (int): out_channels
        kernel_size (int) : kernel_size of Conv2D, defaults to 1.
        strides (int) : strides of Conv2D, defaults to 1.
        use_bias(bool) : whether use bias of Conv2D , if deploy=True, it will be invalid,  defaults to False.
        groups (int) :  groups of Conv2D, defaults to 1. 
        use_bn (bool) : whether use batch norm in module if depoly=None , defaults to True. 
                        if depoly!=None, it will be invalid, 
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in conv_bn blocks, defaults to None.
        deploy (bool): detemine use_bn and use_bias in this group layer if deploy!=None. default to None, 
                       depoly = None => disable re-parameterize attribute (turn off switch_to_deploy)
                       depoly = True => to use deployment config(remove bn), only conv layer will be bulit
                       depoly = False => to use training config() , 
                                enable switch_to_deploy that can convert trained weights(conv_kernel_bias+bn) to deployed weights (only conv_kernel)
        name (str) :'Conv2D'


    Note:

        - PyTorch Conv2d kernel : [out_channels, in_channels//groups, filter_height, filter_width]
        - TensorFlow Conv2d kernel : [filter_height, filter_width, in_channels//groups, out_channels]

        layer.weights = List[ <AutoCastVariable 'Conv2D_BN_Test/conv/kernel:0' shape=(3, 3, 64, 128) dtype=float32, numpy=[] >,
                            <AutoCastVariable 'Conv2D_BN_Test/conv/bias:0' shape=(128,) dtype=float32, numpy= [] >,
                            <tf.Variable 'Conv2D_BN_Test/bn/gamma:0' shape=(128,) dtype=float32, numpy=[] >,
                            <tf.Variable 'Conv2D_BN_Test/bn/beta:0' shape=(128,) dtype=float32, numpy=[] >,
                            <tf.Variable 'Conv2D_BN_Test/bn/moving_mean:0' shape=(128,) dtype=float32, numpy=[] >,
                            <tf.Variable 'Conv2D_BN_Test/bn/moving_variance:0' shape=(128,) dtype=float32, numpy=[]>
        ], here Conv2D_BN_Test is given layer name inherited from 'tf.keras.layers.Layer',  conv and bn are sublayer'names in Conv2D_BN layer
        'Conv2D_BN/conv/kernel:0' shape=(3, 3, 64, 128) which means that kernel size is (3,3), input channel=64 and output_channel(conv's filters)=128

        - deploy=False :  use_bn=True, 
        - deploy=True :   use_bn=False, use_bias=True
        - deploy=None : 
   
    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    x = Input(shape=(256,256,3))
    out = Conv2D_BN(filters = 128, reparam_deploy = False, name = 'Conv2D_BN')(x)
    model = Model(x, out)
    
    print( model.get_layer('Conv2D_BN').weights[0][:,:,0,0]) #before fuse bn+conv

    for layer in model.layers:
        if hasattr(layer,'switch_to_deploy')
            layer.switch_to_deploy()
            print('---------------------')
            for weight in layer.weights:
                print(weight.name)

    print( model.get_layer('Conv2D_BN').weights[0][:,:,0,0]) #after fuse bn+conv
    """
    def __init__(self,
            filters : int,
            kernel_size : int=1,
            strides: int =1,
            use_bias:bool= False,
            groups: int =1,
            use_bn : bool= True,
            bn_epsilon : float= 1e-5,
            bn_momentum : float= 0.9,
            activation : Optional[str] = None,
            deploy : Optional[bool] = None,
            name : str='Conv2D', **kwargs):
        super().__init__(name=name)

        if isinstance(kernel_size, (list, tuple)) :
            self.kernel_size = kernel_size
        else :
            self.kernel_size = (kernel_size, kernel_size)

        self.deploy = deploy
        self.filters = filters
        self.strides = strides
        self.groups = groups
        self.use_bias = use_bias
        self.use_bn = use_bn

        #self.use_bn = False if self.deploy==True else True
        #self.use_bn = use_bn  #use_bn
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum

        if not isinstance(activation,(str, type(None))):
            raise TypeError("activation must be 'str' type like 'relu'"
                        f"but got {type(activation)} ")
    
        self.act_name = activation
        

    def build(self, input_shape):

        _, _, _, self.in_channels = input_shape

        self.use_bn = self.use_bn if self.deploy==None else (not self.deploy)
        self.use_bias = True if self.deploy else self.use_bias
        
        padding = (self.kernel_size[0]//2, self.kernel_size[1]//2)
        if max(padding)>0:
            self.zero_padding2D = ZeroPadding2D(padding=padding,name='pad')

        self.conv = Conv2D(self.filters,
                kernel_size=self.kernel_size,
                strides =self.strides,
                use_bias= self.use_bias , 
                groups = self.groups,
                padding="valid",
                name="conv")
        
        if self.use_bn:
            self.bn = BatchNormalization(epsilon=self.bn_epsilon,
                                    momentum=self.bn_momentum,
                                    name="bn")
        if self.act_name is not None  :
            self.act = Activation(self.act_name, name=self.act_name)

    def call(self,x):

        if hasattr(self,'zero_padding2D'):
            x = self.zero_padding2D(x)

        x = self.conv(x)

        if hasattr(self,'bn'):    
            x = self.bn(x)

        if hasattr(self,'act'): 
            x = self.act(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
                {
                    "in_channels" : self.in_channels,
                    "filters": self.filters,
                    "kernel_size": self.kernel_size,
                    "strides": self.strides,
                    "use_bias": self.use_bias,
                    "groups": self.groups,
                    "use_bn" : self.use_bn,
                    "bn_epsilon" : self.bn_epsilon,
                    "bn_momentum" : self.bn_momentum,
                    "reparam_deploy" : self.deploy,
                    "activation" : self.act.name if hasattr(self, 'act') else None,
                }
        )
        return config

    def _fuse_bn_layer(self)->Tuple[Tensor,Tensor]:
        '''
        conv + bn => fused_conv
        std = tf.sqrt(bn.moving_variance + bn.eps)
        fused_conv.kernel  :  conv.kernel * (bn.gamma / std)
        fused_conv.bias  :  bn.beta - ( bn.moving_mean * bn.gamma / std )

        Note : 
            
            weights shape -
            filters = out_channels

            conv.kernel  : (in_channels, 3, 3, filters)
            conv.bias  : (filters,)
            bn.gamma  : (filters,)
            bn.beta  : (filters,)
            bn.bias moving_mean : (filters,)
            bn.moving_variance  : (filters,)
            [kernel, bias ]= model.get_layer('Conv2D_BN').weights ->List[Tensor]

        '''
        if self.deploy!=False:
            raise ValueError("' to extract original weights(trainning) to fuse bn layer as single one fused_conv,"
                             f"depoly must be False @{self.__class__.__name__}")
    
        # if not hasattr(self,'bn'):
        #     return self.conv.kernel, self.conv.bias
        if not hasattr(self,'bn'):
            return self.conv.weights
        
        kernel = self.conv.kernel
        running_mean =  self.bn.moving_mean
        running_var =  self.bn.moving_variance
        gamma =  self.bn.gamma
        beta =  self.bn.beta
        eps =  self.bn.epsilon

        'kernel'
        std = tf.sqrt(running_var + eps)
        t = gamma / std
        new_kernel = kernel * t
        'bias'
        bias = self.conv.bias if self.use_bias else 0.
        new_bias = beta +(bias- running_mean) * gamma / std 

        #return [kernel * t,  beta - running_mean * gamma / std] 
        return [new_kernel, new_bias]
    
    def weights_convert_to_deploy(self) ->Tuple[Tensor,Tensor]:
        if self.deploy!=False:
            raise ValueError("' to extract original weights(trainning) to fuse bn layer as single one fused_conv,"
                             f"depoly must be False @{self.__class__.__name__}")

        #kernel, bias = self._fuse_bn_layer() 
        weights_map = dict()
        #weights_map['conv'] = [kernel, bias]
        weights_map['conv'] = self._fuse_bn_layer() 
        return  weights_map
    
    def switch_to_deploy(self):
        '''switch_to_deploy
        reparameterize weights inplane
        PyTorch Conv2d kernel :[out_channels, in_channels, filter_height, filter_width]
        TensorFlow Conv2d kernel : [filter_height, filter_width, in_channels, out_channels]
        '''
        if self.deploy or self.deploy==None:
            return
        
        'get new weights by fuse algo. conv+bn => conv'
        fused_weights = self._fuse_bn_layer()  #weights = [kernel, bias]

        #'init a new conv instead of original self.conv'
        'delete self.bn for deployment'
        if hasattr(self,'bn'): 
            self.__delattr__('bn')

        self.built = False
        self.deploy = True
        'build new conv by seting input shape'
        # self.conv.build((1,32,32,self.in_channels))
        super().__call__(tf.random.uniform(shape=(1,32,32,self.in_channels)))

        'update fused_weights to self.conv'
        #print("reparametrized weights :",fused_weights[0][:,:,0,0])
        self.conv.set_weights(fused_weights) 

    

#---------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------
class DepthwiseConv2D_BN(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r""" DepthwiseConv2D_BN (DepthwiseConv2D_no_bias)
    DepthwiseConv2D_BN Module = DepthwiseConv2D + BN + activation

    support fuse depthwise_conv + bn as one conv layer


    References:
        - [Based on @hoangthang1607's  implementation of repvgg] (https://github.com/hoangthang1607/RepVGG-Tensorflow-2/blob/main/repvgg.py)
        - [Inspired by  @model_surgery's model_surgery] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/model_surgery/model_surgery.py)
        - [Inspired by  @notplus's MobileOne-TF2] (https://zhuanlan.zhihu.com/p/543101751) (https://github.com/notplus/MobileOne-TF2/blob/main/mobileone.py)

    Args:
        kernel_size (int) : kernel_size of DWConv2D, defaults to 1.
        strides (int) : strides of DWConv2D, defaults to 1.
        use_bias(bool) : whether use bias of DWConv2D, defaults to False. 
                        but when deploy=True, it will be invalid.
        use_bn (bool) : whether use batch norm in module if depoly=None , defaults to True. 
                        but if depoly!=None, it will be invalid.
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in conv_bn blocks, defaults to None.
        depoly (bool): default to None, 
                       depoly = None is diable re-parameterize attribute (turn off switch_to_deploy), 
                       depoly = True is to use deployment config, layer will be bulit by only conv  
                       depoly = False is to use training config , here 
                                if depoly=False, switch_to_deploy can convert trained weights(conv_kernel+conv_bias+bn) to deployed weights (only conv_kernel)
        name : str='DWConv2D_BN'

    Note:
        - weights 
            layer.weights = List[ <AutoCastVariable 'DWConv2D_Test/conv/kernel:0' shape=(3, 3, 64, 1) dtype=float32, numpy=[] >,
                                <AutoCastVariable 'DWConv2D_Test/conv/bias:0' shape=(64,) dtype=float32, numpy= [] >,
                                <tf.Variable 'DWConv2D_Test/bn/gamma:0' shape=(64,) dtype=float32, numpy=[] >,
                                <tf.Variable 'DWConv2D_Test/bn/beta:0' shape=(64,) dtype=float32, numpy=[] >,
                                <tf.Variable 'DWConv2D_Test/bn/moving_mean:0' shape=(64,) dtype=float32, numpy=[] >,
                                <tf.Variable 'DWConv2D_Test/bn/moving_variance:0' shape=(64,) dtype=float32, numpy=[]>
            ], here DWConv2D_Test is given layer name from class init,  dw_conv and bn are sublayer'names in DepthwiseConv2D_BN class
            'DWConv2D_Test/dw_conv/depthwise_kernel:0' shape=(3, 3, 64, 1) which means that
            kernel size is (3,3), input channel=64

        - deploy=False =>  use_bn=True
        - deploy=True  =>   use_bn=False, use_bias=True
        - deploy=None  =>  use_bn, use_bias 

    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    
    x = Input(shape=(256,256,3))
    out = DepthwiseConv2D_BN(
                        kernel_size=3,
                        strides=1,
                        use_bias = False,
                        use_bn = True,
                        bn_epsilon = 1e-5,
                        bn_momentum = 0.9,
                        activation = None,
                        deploy = False,
                        name = 'DWConv2D_BN_Test')(x)
    model = Model(x, out)
     
    print( model.get_layer('DWConv2D_BN_Test').weights[0][:,:,0,0]) #before fuse bn+conv 

    # switch_to_deploy
    for layer in model.layers:
        if hasattr(layer,'switch_to_deploy')
            layer.switch_to_deploy()
            print('---------------------')
            for weight in layer.weights:
                print(weight.name)
    print( model.get_layer('DWConv2D_BN_Test').weights[0][:,:,0,0]) #after fuse bn+conv
        

    """

    def __init__(self,
            kernel_size : int=1,
            strides: int =1,
            use_bias:bool= False,
            use_bn : bool= True,
            bn_epsilon : float= 1e-5,
            bn_momentum : float= 0.9,
            activation = None,
            deploy : Optional[bool] = False,
            name : str='dw_Conv2D', **kwargs):

        super().__init__(name=name)

        if isinstance(kernel_size, (list, tuple)) :
            self.kernel_size = kernel_size
        else :
            self.kernel_size = (kernel_size, kernel_size)


        if not isinstance(activation,(str, type(None))):
            raise TypeError("activation must be 'str' type like 'relu'"
                         f"but got {type(activation)} @{self.__class__.__name__}")
        
        self.deploy = deploy
        self.strides = strides
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.act_name = activation

    def build(self, input_shape):
        _, _, _, self.in_channels = input_shape
        padding = (self.kernel_size[0]//2, self.kernel_size[1]//2)

        self.use_bn = self.use_bn if self.deploy==None else (not self.deploy)
        self.use_bias = True if self.deploy else self.use_bias
        
        if max(padding)>0:
            self.zero_padding2D = ZeroPadding2D(padding=padding, name='pad')

        self.dw_conv = DepthwiseConv2D(kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        use_bias= self.use_bias,
                                        padding="valid",
                                        name="dw_conv")
        
        if self.use_bn:
            self.bn = BatchNormalization(epsilon=self.bn_epsilon,
                                        momentum=self.bn_momentum,
                                        name="bn")
            
        if self.act_name is not None  :
            self.act = Activation(self.act_name, name=self.act_name)
      
    def call(self,x):

        if hasattr(self,'zero_padding2D'):
            x = self.zero_padding2D(x)
        
        x = self.dw_conv(x)   
        
        if hasattr(self,'bn'): 
            x = self.bn(x)

        if hasattr(self,'act'): 
            x = self.act(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
                {
                "reparam_deploy" : self.deploy,
                "in_channels" : self.in_channels,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "use_bias": self.use_bias,
                "use_bn" : self.use_bn,
                "bn_epsilon" : self.bn_epsilon,
                "bn_momentum" : self.bn_momentum,
                "activation" : self.act.name if hasattr(self,'act') else None,
                }
        )
        return config
    
    def _fuse_bn_layer(self)->Tuple[Tensor,Tensor]:
        '''
        conv + bn => fused_conv
        std = tf.sqrt(bn.moving_variance + bn.eps)
        fused_conv.kernel  :  conv.kernel * (bn.gamma / std)
        #fused_conv.bias  :  bn.beta - ( bn.moving_mean-conv.bias) * bn.gamma / std )
        fused_conv.bias  :  bn.beta + (conv.bias - bn.moving_mean)* bn.gamma/std

        Note : 
            
            weights shape -
            dw_conv.depthwise_kernel  : (3, 3, in_channels, 1)
            dw_conv.bias  : (in_channels,)
            bn.gamma  : (in_channels,)
            bn.beta  : (in_channels,)
            bn.bias moving_mean : (in_channels,)
            bn.moving_variance  : (in_channels,)

            [kernel, bias ]= model.get_layer('DepthwiseConv2D_BN').weights ->List[Tensor]

        '''
        if self.deploy!=False:
            raise ValueError("' to extract original weights(trainning) to fuse bn layer as single one fused_conv,"
                             f"depoly must be False @{self.__class__.__name__}")
    
        if not hasattr(self,'bn'):
            ' no fuse, just return original dw_conv.weights '
            return self.dw_conv.weights
        
        kernel = self.dw_conv.depthwise_kernel
        '(3, 3, 64, 1)'
        kernel = tf.transpose(kernel,[0,1,3,2]) # (3, 3, 64, 1) => (3, 3, 1, 64)

        running_mean =  self.bn.moving_mean
        running_var =  self.bn.moving_variance
        gamma =  self.bn.gamma
        beta =  self.bn.beta
        eps =  self.bn.epsilon

        'conv kernel'
        std = tf.sqrt(running_var + eps)
        t = gamma / std
        new_kernel = tf.transpose(kernel*t,[0,1,3,2]) #kernel * t

        'conv bias'
        bias = self.dw_conv.bias if self.use_bias else 0.
        new_bias = beta +(bias- running_mean) * gamma / std 
        return [new_kernel, new_bias]
    
    def weights_convert_to_deploy(self) ->Tuple[Tensor,Tensor]:
        weights = self._fuse_bn_layer() 
        weights_map = dict()
        weights_map['dw_conv'] = weights
        return  weights_map
    
    def switch_to_deploy(self):

        if self.deploy or self.deploy is None:
            return
        
        'fuse algo. dw_conv+bn => dw_conv'
        fused_weights = self._fuse_bn_layer()  #weights = [kernel, bias]

        'delete self.bn '
        if hasattr(self,'bn'): 
            self.__delattr__('bn')
    
        're build dw_conv_bn by seting input shape/ deploy = True / built = False'
        self.built = False
        self.deploy = True
        super().__call__(tf.random.uniform(shape=(1,32,32,self.in_channels)))

        'update fused_weights to self.conv'
        self.dw_conv.set_weights(fused_weights) 


#---------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------
class SeparableConv2D_BN(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r""" SeparableConv2D_BN (DepthwiseSeparableConvModule)

    SeparableConv2D_BN = (DepthwiseConv2D + BN + ACT) +  (Conv2D + BN + ACT),
    where we also call 2nd conv as pointwise conv (pw_conv) that has strides=1 and kernel_size=1
                       
    Return :
           tf.Tensor

    Note : 
        - support fuse conv+bn as one conv layer by switch_to_deploy, but need some experiments to verify

    References:
        - [Based on implement of DepthwiseSeparableConvModule @mmcv] (https://github.com/open-mmlab/mmcv/blob/main/mmcv/cnn/bricks/depthwise_separable_conv_module.py)
        - [more details in paper]  (https://arxiv.org/pdf/1704.04861.pdf )

    Args:
        filters (int):  Number of out_channels produced by the convolution. 
                    Same as that in `tf.nn.conv2d`. if filters=-1, out_channels will be same as in_channels
        kernel_size (int) :  Size of the convolving kernel of DepthwiseConv, defaults to 1.
                    Same as that in `tf.nn.depthwise_conv2d`
        strides (int) : Stride of the depthwise convolution. defaults to 1. 
                    Same as that in `tf.nn.depthwise_conv2d`
        use_bias (bool) : whether use convolving bias for  DepthwiseConvModule and ConvModule, default to False
                    Same as that in `tf.nn.depthwise_conv2d` and `tf.nn.conv2d`
                    if deploy=True, it will be invalid.
        use_bn(bool) : whether use batch norm after convolution ops, default to True
                    if depoly!=None, it will be invalid.      
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in DepthwiseConvModule and ConvModule defaults to None.
        depoly (bool): determine depolyment config . default to None, 
                       depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfg depend on above args.
                       depoly = True => to use deployment config, only conv layer will be bulit
                       depoly = False => to use training config() , 
                                if depoly=False, switch_to_deploy can convert trained weights(conv_kernel+conv_bias+bn) to deployed weights (only conv_kernel)
        name (str) :'SeparableConv2D_BN'
    

    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    x = Input(shape=(256,256,64))
    out = SeparableConv2D_BN(
                        filters=128,
                        kernel_size=3,
                        strides=1,
                        use_bias = False,
                        use_bn = True,
                        bn_epsilon = 1e-5,
                        bn_momentum = 0.9,
                        activation = None,
                        deploy = False,
                        name = 'SeparableConv2D_BN')(x)
    model = Model(x, out)
    model.summary(100)

    """
    def __init__(self,
            filters : int,
            kernel_size : int=3,
            strides: int =1,
            use_bias:bool= False,
            use_bn : bool= True,
            bn_epsilon : float= 1e-5,
            bn_momentum : float= 0.9,
            activation : Optional[str] = None,
            deploy : Optional[bool] = None,
            name : str='Conv2D', **kwargs):
        super().__init__(name=name)
      
        if isinstance(kernel_size, (list, tuple)) :
            self.kernel_size = kernel_size
        else :
            self.kernel_size = (kernel_size, kernel_size)



        if not isinstance(activation,(str, type(None))):
            raise TypeError("dw_activation must be 'str' type like 'relu'"
                            f"but got {type(activation)} ")

        self.deploy = deploy
        self.filters = filters
        self.strides = strides
        self.use_bias = use_bias

        self.use_bn = use_bn
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
 
        self.act_name = activation
  
 

    def build(self, input_shape):

        _, h, w, self.in_channels = input_shape
        if self.filters < 0:
            self.filters =  self.in_channels

        self.dw_conv = DepthwiseConv2D_BN(
                                        kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        use_bias = self.use_bias,
                                        use_bn = self.use_bn, 
                                        bn_epsilon = self.bn_epsilon,
                                        bn_momentum = self.bn_momentum,
                                        activation = self.act_name,
                                        deploy = self.deploy,
                                        name = 'dw_conv')
        
        self.pw_conv = Conv2D_BN(filters = self.filters,
                                kernel_size=1,
                                strides=1,
                                use_bias = self.use_bias,
                                use_bn = self.use_bn, 
                                bn_epsilon = self.bn_epsilon,
                                bn_momentum = self.bn_momentum,
                                activation = self.act_name,
                                deploy = self.deploy,
                                name = f'pw_conv')   
        
     
    def call(self,x) ->tf.Tensor:
        'depthwise_conv'
        x = self.dw_conv(x)
        'pointwise_conv'
        x = self.pw_conv(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
                {
                    "filters": self.filters,
                    "kernel_size": self.kernel_size,
                    "strides": self.strides,
                    "use_bias": self.use_bias,
                    "use_bn": self.use_bn,                     
                    "bn_epsilon" : self.bn_epsilon,
                    "bn_momentum" : self.bn_momentum,
                    "reparam_deploy" : self.deploy,
                    "activation" : self.act_name,
                }
        )
        return config
    
    def switch_to_deploy(self):

        if self.deploy or self.deploy==None:
            return
        
        self.dw_conv.switch_to_deploy()
        dw_conv_weights  = self.dw_conv.weights
        #print(dw_conv_weights[0][:,:,0,0])

        self.pw_conv.switch_to_deploy()
        pw_conv_weights  = self.pw_conv.weights
       
        're build dw_conv_bn by seting input shape/ deploy = True / built = False'
        self.built = False
        self.deploy = True       
        super().__call__(tf.random.uniform(shape=(1,32,32,self.in_channels)))

        'update fused_weights to self.conv'
        self.dw_conv.set_weights(dw_conv_weights) 
        self.pw_conv.set_weights(pw_conv_weights)     


#---------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------
class GhostConv2D(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r""" GhostConv2D (ghost_module)

    A Light-weight convolutional neural networks (CNNs) are specially designed for
    applications on mobile devices with faster inference speed.
    decoupled fully connected attention (dfc) used in moddule is a hardware-friendly attention mechanism (
    it was appiled on GhostNetV2)

    GhostConv2D was used in GhostNetV2/GhostNetV1/GhostBottoleNeck

    Return :
           tf.Tensor

    Architecture :
        in: (b,40,40,128) => GhostBottleNeck => out: (b,40,40,256)

        GhostBottleNeck_cfg = {
            filters = 256,
            kernel_size = 1, 
            dw_kernel_size =3,
            strides = 1,    
            use_depthwise= True,
            use_dfc_attention = False,
        }

        #[from, number, module, args]
        [-1, 1, Conv, [128, 1, 1]]              #arbitrary dummy input with shape (80,80,128)
        -------------------------------------------------------------------------------------
        [-1, 1, dfc_block, [256, 3, 1]],        # (80,80,128)=>(80,80,256) 
        [-2, 1, Conv2D_BN, [128, 1, 1]],        # (80,80,128)=>(80,80,128)
        [-1, 1, DWConv2D_BN, [128, 3, 1]],      # (80,80,128)=>(80,80,128)
        [[-1, -2,], 1, Concat, [axis=-1]],      # (80,80,128*2)
        [-1, -4, MUL, []],                      # (80,80,256)*(80,80,256)=>(80,80,256)


    Note : 
        - this conv module was implemented by Conv2D_BN and DepthwiseConv2D_BN
        - support fuse conv+bn as one conv layer by switch_to_deploy, but need some experiments to verify
        - although dfc attention can support strides=2, strides should be always 1 in gerneral
          ghostnet used another conv with stride=2 to do downsample in ghost-bottoleneck. 
    

    References:
        - [GhostNetV2  paper] (https://arxiv.org/pdf/2211.12905.pdf)
        - [Based on GhostNetV2 implement @likyoo] (https://github.com/likyoo/GhostNetV2-PyTorch/blob/main/ghostnetv2.py)
        - [Inspired by @ultralytics's GhostConv torch module] (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py)
        - [Inspired by @leondgarse's ghost_module] (https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/ghostnet/ghostnet_v2.py)
    
    Args:
        filters (int): out_channels.
        kernel_size (int) :  kernel of primary conv, defaults to 1.
        dw_kernel_size (int) :  kernel of cheap conv, defaults to 3
                              in general, dw_kernel_size should be always is 3.
        strides (int) : stride of primary conv. defaults to 1. 
                        in general, strides should be always is 1 . 
        use_depthwise (bool) : whether use DepthwiseConv2D for cheap conv , default to True.
                               if not, cheap_conv will apply Conv2D(groups=in_channels),
        use_dfc_attention (bool) : whether use dfc attention block for residual branch, default to False
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) = activation used in conv_bn blocks, defaults to None.
        depoly (bool): determine depolyment config . default to None, 
                       depoly = None => diable re-parameterize attribute (turn off switch_to_deploy), all cfg depend on above args.
                       depoly = True => to use deployment config, only conv layer will be bulit
                       depoly = False => to use training config() , 
                                if depoly=False, switch_to_deploy can convert trained weights(conv_kernel+conv_bias+bn) to deployed weights (only conv_kernel)
        name (str) :'GhostConv2D'
    
    Note:

    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    x = Input(shape=(256,256,16))
    out = GhostConv2D(filters = 16,
                    kernel_size =1,
                    dw_kernel_size =3,
                    strides=1,
                    use_depthwise = True,
                    use_dfc_attention = False,
                    bn_epsilon = 1e-5,
                    bn_momentum = 0.9,
                    activation = 'relu',
                    deploy = False,
                    name = 'GhostConv2D')(x)
    model = Model(x, out)
    model.summary(100)


    """


    def __init__(self,
            filters : int,
            kernel_size : int=1,
            dw_kernel_size : int=3,
            strides: int =1,
            use_depthwise : bool = True,
            use_dfc_attention : bool = False,
            bn_epsilon : float= 1e-5,
            bn_momentum : float= 0.9,
            activation : Optional[str] = 'relu',
            deploy : Optional[bool] = None,
            name : str='Conv2D', **kwargs):
        super().__init__(name=name)
      
        if isinstance(kernel_size, (list, tuple)) :
            self.kernel_size = kernel_size
        else :
            self.kernel_size = (kernel_size, kernel_size)

        if isinstance(dw_kernel_size, (list, tuple)) :
            self.dw_kernel_size = dw_kernel_size
        else :
            self.dw_kernel_size = (dw_kernel_size, dw_kernel_size)

        if not isinstance(activation,(str, type(None))):
            raise TypeError("activation must be 'str' type like 'relu'"
                        f"but got {type(activation)} ")

        self.deploy = deploy
        self.filters = filters
        self.strides = strides
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.use_depthwise = use_depthwise
        self.use_dfc_attention = use_dfc_attention

        self.act_name = activation
        self.deploy = deploy


    def build(self, input_shape):

        _, h, w, c = input_shape
        if self.filters < 0:
            self.filters =  c

        self.hidden_channels = int(math.ceil(float(self.filters) / 2.))
        #self.hidden_channels = self.filters//2

        self.primary_conv = Conv2D_BN(filters = self.hidden_channels,
                                        kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        bn_epsilon = self.bn_epsilon,
                                        bn_momentum = self.bn_momentum,
                                        activation = self.act_name,
                                        deploy = self.deploy,
                                        name = f'primary_conv')
        
        if self.use_depthwise:
            self.cheap_conv = DepthwiseConv2D_BN(
                                        kernel_size=self.dw_kernel_size,
                                        strides=1,
                                        bn_epsilon = self.bn_epsilon,
                                        bn_momentum = self.bn_momentum,
                                        activation = self.act_name,
                                        deploy = self.deploy,
                                        name = 'cheap_conv')
        else:
            self.cheap_conv = Conv2D_BN(filters = self.hidden_channels,
                                        kernel_size=self.dw_kernel_size,
                                        strides=1,
                                        groups = self.hidden_channels,
                                        bn_epsilon = self.bn_epsilon,
                                        bn_momentum = self.bn_momentum,
                                        activation = self.act_name,
                                        deploy = self.deploy,
                                        name = f'cheap_conv')   
        
        self.concat = Concatenate(axis=-1, name='concat')

        if self.use_dfc_attention :
            ' decoupled_fully_connected_attention used in ghostnetv2'
            self.dfc_attn = tf.keras.Sequential(
                [
                    AveragePooling2D(pool_size=(2, 2), 
                                    strides=2, 
                                    padding='valid', 
                                    name='attn_AP'
                    ),
                    Conv2D_BN(filters = self.filters,
                            kernel_size = self.kernel_size,
                            strides=self.strides,
                            bn_epsilon = self.bn_epsilon,
                            bn_momentum = self.bn_momentum,
                            name = f'attn_1'
                    ),
                    DepthwiseConv2D_BN(kernel_size=(1,5),
                                        bn_epsilon = self.bn_epsilon,
                                        bn_momentum = self.bn_momentum,
                                        name = 'attn_2'
                    ),
                    DepthwiseConv2D_BN(kernel_size=(5,1),
                                        bn_epsilon = self.bn_epsilon,
                                        bn_momentum = self.bn_momentum,
                                        name = 'attn_3'
                    ),
                    tf.keras.layers.Lambda(
                        lambda x: tf.keras.activations.sigmoid(x)
                    ),
                    Resizing( h//self.strides, w//self.strides, interpolation="nearest", name='attn_resize')

                ],
                name="dfc_attn_block",
            )
            self.mul = Multiply(name='mul')

    def call(self,inputs) ->tf.Tensor:
        'GhostModule_v1'
        x = self.primary_conv(inputs)
        ghost_out = self.concat([x, self.cheap_conv(x)])
        'GhostModule_v2 (add dfc_attention)'
        if self.use_dfc_attention:
           #ghost_res = self.dfc_attn(inputs)
           ghost_out =  self.mul([ghost_out, self.dfc_attn(inputs)])
        
        return ghost_out

    def get_config(self):
        config = super().get_config()
        config.update(
                {
                    "filters": self.filters,
                    "kernel_size": self.kernel_size,
                    "dw_kernel_size": self.dw_kernel_size,
                    "strides": self.strides,
                    "use_dfc_attention" : self.use_dfc_attention,
                    "use_depthwise" : self.use_depthwise,   
                    "bn_epsilon" : self.bn_epsilon,
                    "bn_momentum" : self.bn_momentum,
                    "reparam_deploy" : self.deploy,
                    "activation" : self.act.name if hasattr(self,'act') else None,
                }
        )
        return config
    
    def switch_to_deploy(self):

        if self.deploy or self.deploy==None:
            return
        
        'get fused weight of conv_bn_1 and remove its bn layer'
        self.primary_conv.switch_to_deploy()
        self.cheap_conv.switch_to_deploy()



#---------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------
class Conv2DTranspose_BN(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r""" Conv2DTranspose_BN_Layer (conv2d_no_bias)


    """
    def __init__(self,
            filters : int,
            kernel_size : int=1,
            strides: int = 2,
            use_bias:bool= False,
            use_bn : bool= True,
            bn_epsilon : float= 1e-5,
            bn_momentum : float= 0.9,
            activation : Optional[str] = None,
            name : str='Conv2D', **kwargs):
        super().__init__(name=name)

        if isinstance(kernel_size, (list, tuple)) :
            self.kernel_size = kernel_size
        else :
            self.kernel_size = (kernel_size, kernel_size)

        self.filters = filters #deconv_filters
        self.strides = strides
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum

        if not isinstance(activation,(str, type(None))):
            raise TypeError(
                "activation must be 'str' type like 'relu'"
                f"but got {type(activation)} "
            )
    
        self.act_name = activation
        
    def build(self, input_shape):

        _, _, _, self.in_channels = input_shape

        self.trans_conv = Conv2DTranspose(
            filters = self.filters, 
            kernel_size = self.kernel_size, 
            strides = self.strides, 
            padding="same",
            use_bias=self.use_bias,
            name= f'TransConv'
        )
        if self.use_bn:
            self.bn = BatchNormalization(
                epsilon=self.bn_epsilon,
                momentum=self.bn_momentum,
                name="bn"
            )
        if self.act_name is not None  :
            self.act = Activation(
                self.act_name, name=self.act_name
            )

    def call(self,x):

        x = self.trans_conv(x)
        if hasattr(self,'bn'):    
            x = self.bn(x)
        if hasattr(self,'act'): 
            x = self.act(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels" : self.in_channels,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "use_bias": self.use_bias,
                "use_bn" : self.use_bn,
                "bn_epsilon" : self.bn_epsilon,
                "bn_momentum" : self.bn_momentum,
                "activation" : self.act.name if hasattr(self, 'act') else None,
            }
        )
        return config
    