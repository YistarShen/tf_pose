
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import BatchNormalization,Activation
from lib.layers import Conv2D_BN, DepthwiseConv2D_BN

class RepVGGConv2D(tf.keras.layers.Layer):
    VERSION = '1.0.0'
    r""" Conv2D_Layer (conv2d_no_bias)
   

    References:
        - [Inspired by @hoangthang1607's  implementation of repvgg] (https://github.com/hoangthang1607/RepVGG-Tensorflow-2/blob/main/repvgg.py)
        - [Inspired by  @notplus's MobileOne-TF2] (https://zhuanlan.zhihu.com/p/543101751) (https://github.com/notplus/MobileOne-TF2/blob/main/mobileone.py)

    Args:
        filters (int): out_channels of conv, if use_depthwise=True, it wil be invalid.
        kernel_size (int) : kernel size of primary conv. if kernel_size=1, auxiliary_conv is invliad.  defaults to 3
        use_depthwise (bool) : whether use depthwise conv. if True, filters and groups both are invliad,  default to False
        strides (int) : if true, shorcut branch is invalid. defaults to 1
        use_bias (bool):  whether use conv_bias. defaults to False,
        groups (int) : groups of conv , if use_depthwise=True, it wil be invalid. defaults to 1
        bn_epsilon (float) : defaults to 1e-5,
        bn_momentum (float): defaults to 0.9,
        activation (str) : None,
        depoly (bool) : defaults to False,
        name : str='RepVGGConv2D'

    Note:
        - all weights.name in RepVGGConv2D layer
     
        if deploy==False
            RepVGGConv2D/REPARAM_1/conv/kernel:0
            RepVGGConv2D/REPARAM_1/bn/gamma:0
            RepVGGConv2D/REPARAM_1/bn/beta:0
            RepVGGConv2D/REPARAM_2/conv/kernel:0
            RepVGGConv2D/REPARAM_2/bn/gamma:0
            RepVGGConv2D/REPARAM_2/bn/beta:0
            RepVGGConv2D/REPARAM_1/bn/moving_mean:0
            RepVGGConv2D/REPARAM_1/bn/moving_variance:0
            RepVGGConv2D/REPARAM_2/bn/moving_mean:0
            RepVGGConv2D/REPARAM_2/bn/moving_variance:0

        if deploy==True
            RepVGGConv2D/REPARAM_1/conv/kernel:0
            RepVGGConv2D/REPARAM_1/conv/bias:0

    Examples:

    ```python
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    x = Input(shape=(256,256,32))
    out = RepVGGConv2D(filters = 64,
                kernel_size=3,
                strides=1,
                use_bias = False,
                use_depthwise =False,
                groups=1,
                activation = None,
                deploy = False,
                name = 'RepVGGConv2D')(x)

    model = Model(x, out)
    model.summary(100)

    print( model.get_layer('RepVGGConv2D').weights[0][:,:,0,0])
    for layer in model.layers:
        if layer.name == 'RepVGGConv2D':
            layer.switch_to_deploy()
            print('---------------------')
            for weight in layer.weights:
                print(weight.name)
    model.summary(100)
    print( model.get_layer('RepVGGConv2D').weights[0][:,:,0,0])

    """
    def __init__(self,
            filters : int,
            kernel_size : int=1,
            strides: int =1,
            use_bn_identity : bool = True,
            use_depthwise : bool=False,
            use_bias:bool= False,
            groups: int =1,
            bn_epsilon : float= 1e-5,
            bn_momentum : float= 0.9,
            activation = 'relu',
            deploy : bool = False,
            name : str='Conv2D',
            **kwargs):
        super().__init__(name=name,  **kwargs)
        #assert use_depthwise == False, f'not yet support depthwise conv  @{self.__class__.__name__} '
        # assert kernel_size not in (1,3), \
        # f"this version only support kernel_size=1 or 3  @{self.__class__.__name__}"

        if isinstance(kernel_size, (list, tuple)) :
            self.kernel_size = kernel_size
        else :
            self.kernel_size = (kernel_size, kernel_size)

        if self.kernel_size not in [(3,3),(1,1)]:
            raise TypeError(f"now, only support kernel_size=3 or 1 @{self.__class__.__name__}"
                            f"but got {self.kernel_size} ")

        self.deploy = deploy
        self.use_bn_identity = use_bn_identity
        self.use_depthwise = use_depthwise

        self.filters = filters
        self.strides = strides
        self.use_bias = True if self.deploy else use_bias
        self.groups = groups
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum

        if not isinstance(activation,(str, type(None))):
            raise TypeError("activation must be 'str' type like 'relu'"
                        f"but got {type(activation)} ")
    
        self.act_name = activation
        
    def build(self, input_shape):
        if not self.built and self.deploy:
            print(f'\nRebuild deploy layer : {self.name} .............')

        _,_,_, self.in_channels = input_shape

        'primary conv with kernel_size=3 or 1 '
        if self.use_depthwise:
            self.groups = self.in_channels
            self.filters = self.in_channels

            self.conv_bn_1  = DepthwiseConv2D_BN(
                                    kernel_size=self.kernel_size,
                                    strides=self.strides,
                                    use_bias = True if self.deploy else self.use_bias,
                                    use_bn = True,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum =self.bn_momentum,
                                    activation = None,
                                    deploy = self.deploy,
                                    name = 'REPARAM_1'
            ) 
        else:
            self.conv_bn_1 =  Conv2D_BN(
                                    filters = self.filters,
                                    kernel_size=self.kernel_size,
                                    strides= self.strides,
                                    use_bias = True if self.deploy else self.use_bias,
                                    groups=self.groups,
                                    use_bn = True,
                                    activation = None,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum = self.bn_momentum,
                                    deploy = self.deploy,
                                    name = 'REPARAM_1'
            )  

        if not self.deploy:  
            'auxiliary conv with kernel_size =1 and kernel_size of primary conv is (3,3)'
            if self.kernel_size == (3,3) : 
                if self.use_depthwise :       
                    self.conv_bn_2  = DepthwiseConv2D_BN(
                                            kernel_size=1,
                                            strides=self.strides,
                                            use_bias = self.use_bias,
                                            use_bn = True,
                                            bn_epsilon = self.bn_epsilon,
                                            bn_momentum =self.bn_momentum,
                                            activation = None,
                                            deploy = self.deploy,
                                            name = 'REPARAM_2'
                    )  
                else :
                    self.conv_bn_2 =  Conv2D_BN(filters = self.filters,
                                            kernel_size=1,
                                            strides= self.strides,
                                            use_bias = self.use_bias,
                                            groups=self.groups,
                                            use_bn = True,
                                            activation = None,
                                            bn_epsilon = self.bn_epsilon,
                                            bn_momentum = self.bn_momentum,
                                            deploy = self.deploy,
                                            name = 'REPARAM_2'
                    )
            
            'auxiliary shorcut branch with BatchNormalization if in_channels=filters and strides=1'
            if self.use_bn_identity and self.filters == self.in_channels and self.strides == 1:
                self.bn_identity = BatchNormalization(
                        epsilon=self.bn_epsilon,
                        momentum=self.bn_momentum,
                        name="REPARAM_0_bn"
                )        
            
        if self.act_name is not None  :
            self.act = Activation(self.act_name, name=self.act_name)

    def call(self,inputs):
        
        feats = []
        feats.append(self.conv_bn_1(inputs))
 
        
        if hasattr(self,'bn_identity'):
            id_out = self.bn_identity(inputs)
            feats.append(id_out)

        if hasattr(self,'conv_bn_2'):
            feats.append(self.conv_bn_2(inputs))

        out = tf.keras.layers.Add(name='Add')(feats)

        if hasattr(self,'act'):
            out = self.act(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
                {
                    "in_channels": self.in_channels,
                    "filters": self.filters,
                    "kernel_size": self.kernel_size,
                    "strides": self.strides,
                    "use_bias": self.use_bias,
                    "groups": self.groups,
                    "bn_epsilon" : self.bn_epsilon,
                    "bn_momentum" : self.bn_momentum,
                    "deploy" : self.deploy,
                    "activation" : self.act.name if self.act_name is not None else  None
                }
        )
        return config

            
    def _convert_bn_identity(self, bn_identity):
        '''
        filters = in_channels=out_channels
        conv2d with group=1 , kernel : (ks,ks, filters, fiters)
        conv2d with group=2 , kernel : (ks,ks, filters//2, fiters)
        conv2d with group=in_filters , kernel : (ks,ks, 1, fiters)
        dw_conv2d,  kernel : (ks,ks, fiters, 1)
        '''
        if self.use_depthwise:
            kernel_value = np.zeros(
                        (1, 1, self.conv_bn_1.weights[0].shape[2], 1), dtype=np.float32
            )
            #conv2d,  kernel : (ks,ks, fiters//groups, fiters)
            for i in range(self.in_channels):
                kernel_value[0, 0, i, 0] = 1
            kernel_value = tf.transpose(kernel_value,[0,1,3,2]) # (3, 3, 64, 1) => (3, 3, 1, 64)

        else:
            input_dim = self.in_channels // self.groups 
            ks = 1 if self.groups == 1 or self.kernel_size==(1,1) else 3
            kernel_value = np.zeros(
                    (ks, ks, self.conv_bn_1.weights[0].shape[2], self.conv_bn_1.weights[0].shape[3]), dtype=np.float32
            )
            for i in range(self.in_channels):
                if ks == 1 :
                    kernel_value[0, 0, i%input_dim, i] = 1
                else:
                    kernel_value[1, 1, i%input_dim, i] = 1

        # input_dim = self.in_channels // (self.groups if not self.use_depthwise else self.in_channels)
        # # kernel_value = np.zeros(
        # #             (3, 3, input_dim, self.in_channels), dtype=np.float32
        # # )

        # # for i in range(self.in_channels):
        # #     kernel_value[1, 1, i % input_dim, i] = 1 

        # if self.groups == 1 or self.kernel_size==(1,1) :
        #     ks = 1
        # else:
        #     ks = 3   

        # kernel_value = np.zeros(
        #             (ks, ks, self.conv_bn_1.weights[0].shape[2], self.conv_bn_1.weights[0].shape[3]), dtype=np.float32
        # )
        # '''
        # filters = in_channels=out_channels
        # conv2d with group=1 , kernel : (ks,ks, filters, fiters)
        # conv2d with group=2 , kernel : (ks,ks, filters//2, fiters)
        # conv2d with group=in_filters , kernel : (ks,ks, 1, fiters)
        # dw_conv2d,  kernel : (ks,ks, fiters, 1)
        # '''

    
        # #conv2d,  kernel : (ks,ks, fiters//groups, fiters)
        # for i in range(self.in_channels):
        #     if ks == 1 :
        #         if self.use_depthwise:
        #             kernel_value[0, 0, i, 0] = 1
        #         else:
        #             kernel_value[0, 0, i % input_dim, i] = 1
        #     else:
        #         kernel_value[1, 1, i if self.use_depthwise else i % input_dim, 0 if self.use_depthwise else i]  
        #         #kernel_value[1, 1, i % input_dim, i] = 1
                
        self.id_tensor = tf.convert_to_tensor(
                    kernel_value, dtype=np.float32
        )

        
        kernel = self.id_tensor
        running_mean = bn_identity.moving_mean
        running_var = bn_identity.moving_variance
        gamma = bn_identity.gamma
        beta = bn_identity.beta
        eps = bn_identity.epsilon
        
        std = tf.sqrt(running_var + eps)
        t = gamma / std

        new_bias =  beta - running_mean * gamma / std
        new_kernel = tf.transpose(kernel*t,[0,1,3,2]) if self.use_depthwise else kernel*t

        #return [kernel * t, beta - running_mean * gamma / std]
        return [new_kernel, new_bias]
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            if self.strides == 2:
                return tf.pad(kernel1x1, tf.constant([[0, 2], [0, 2], [0, 0], [0, 0]]), "CONSTANT")
            else:
                return tf.pad(kernel1x1, tf.constant([[1, 1], [1, 1], [0, 0], [0, 0]]), "CONSTANT")
           

    def switch_to_deploy(self):

        if self.deploy :
            return
        
        'get equivalent kernel and bias by fuse conv_bn_1 + conv_bn_2 + bn_identity => conv_bn_1 without bn'

        'get fused weight of conv_bn_1 and remove its bn layer'
        self.conv_bn_1.switch_to_deploy()
        kernel3x3  , bias3x3  = self.conv_bn_1.weights

        'get fused weight of conv_bn_1 and remove its bn layer'
        if hasattr(self,'conv_bn_2'):
            self.conv_bn_2.switch_to_deploy()
            kernel1x1, bias1x1 = self.conv_bn_2.weights
            self.__delattr__('conv_bn_2')
        else:
            kernel1x1, bias1x1 = None, 0.

        'get trasformed weight of bn_identity and remove it self'
        if hasattr(self,'bn_identity'):
            kernelid, biasid = self._convert_bn_identity(self.bn_identity)
            self.__delattr__('bn_identity')
        else:
            kernelid, biasid = 0., 0.
        #print(kernel3x3.shape,kernelid.shape)
        'get new weight (kernel,bias) by (kernelid, kernel1x1, kernel3x3) and (biasid,bias1x1,bias3x3)'
        kernel = kernel3x3  + kernelid + self._pad_1x1_to_3x3_tensor(kernel1x1)
        bias = bias3x3 + bias1x1 + biasid
       
        're-bulid layer and set self.deploy=True '
        self.built = False
        self.deploy = True 
        super().__call__(tf.random.uniform(shape=(1,32,32,self.in_channels))) 

        'update new weight for conv_bn_1.conv , bn layer in conv_bn_1 already was removed'
        if not self.use_depthwise:
            self.conv_bn_1.conv.set_weights([kernel,bias])
        else:    
            self.conv_bn_1.dw_conv.set_weights([kernel,bias])