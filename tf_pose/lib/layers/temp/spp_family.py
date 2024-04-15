'Spatial Pyramid Pooling Family'
from typing import Optional, Tuple, Union,Sequence
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import ZeroPadding1D, ZeroPadding2D, Concatenate, Multiply, Reshape
from tensorflow import Tensor
import tensorflow as tf
from ..base_conv import Conv2D_BN

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
class SPPCSPC(tf.keras.layers.Layer):
    VESRION = '1.0.0'
    r"""SPPCSPC_Module(SpatialPyramidPoolingBottleneck)   
    spatial_pyramid_pooling_fast
    https://blog.csdn.net/weixin_43694096/article/details/126354660
    """
    def __init__(self, 
                out_channels : int,
                expansion : float = 0.5,
                depth : int= 2,
                SPPFCSPC : bool = False,
                lite_type : bool = False,
                pool_sizes =(5, 9, 13),
                bn_epsilon : float= 1e-5,
                bn_momentum : float= 0.9,
                activation = 'relu',
                name='SPPCSPC',
                **kwargs):
        super(SPPCSPC, self).__init__(name=name)
        
        if not isinstance(pool_sizes, (list, tuple)) :
            raise TypeError(f"pool_sizes must be list , tuple, int type @{self.__class__.__name__}"
                            f"but got {type(pool_sizes)}"
                )
    

        if (not isinstance(bn_epsilon,(float))) or (not isinstance(bn_momentum,(float))):
                raise TypeError(f"bn_eps and  bn_momentum must be 'float' type @{self.__class__.__name__}"
                                f"but got eps:{type(bn_epsilon)}, momentum:{type(bn_momentum)}"
                )

        if not isinstance(activation,(str, type(None))):
            raise TypeError("activation must be 'str' type like 'relu'"
                         f"but got {type(activation)} @{self.__class__.__name__}"
            )

        self.out_channels = out_channels
        self.expansion = expansion
        self.hidden_channels = int(2*self.out_channels*self.expansion)
        self.depth = 1 if lite_type else depth
        self.lite_type = True if  self.depth>1 else False
        self.SPPFCSPC = SPPFCSPC
        self.pool_sizes = pool_sizes if not self.SPPFCSPC else (5,5,5)
            
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.act = activation

    def build(self, input_shape):
        #self.hidden_channels = int(2*self.out_channels*self.expansion)

        self.short_conv = Conv2D_BN(filters = self.hidden_channels,
                                    kernel_size=1,
                                    strides=1,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum = self.bn_momentum,
                                    activation = self.act,
                                    name = 'short_Conv')
        
        self.prev_conv = tf.keras.Sequential(layers = [ Conv2D_BN(filters = self.hidden_channels,
                                                                kernel_size =1,
                                                                strides =1,
                                                                bn_epsilon = self.bn_epsilon,
                                                                bn_momentum = self.bn_momentum,
                                                                activation = self.act,
                                                                name = "pre_conv_1")]
                                            ,name='pre_conv_block')
        

        # self.prev_conv = tf.keras.Sequential(layers=[], name='block')
        # self.prev_conv.add(Conv2D_BN(filters = self.hidden_channels,
        #                                                         kernel_size =1,
        #                                                         strides =1,
        #                                                         bn_epsilon = self.bn_epsilon,
        #                                                         bn_momentum = self.bn_momentum,
        #                                                         activation = self.act,
        #                                                         name = "pre_conv_1")
        # )
        
        if self.depth>1:  
            self.prev_conv.add(Conv2D_BN(filters = self.hidden_channels,
                                    kernel_size =3,
                                    strides =1,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum = self.bn_momentum,
                                    activation = self.act,
                                    name = "pre_conv_2")   
            ) 
            self.prev_conv.add(Conv2D_BN(filters = self.hidden_channels,
                                        kernel_size =1,
                                        strides =1,
                                        bn_epsilon = self.bn_epsilon,
                                        bn_momentum = self.bn_momentum,
                                        activation = self.act,
                                        name = "pre_conv_3")    
            ) 


        self.MaxPooling2D_list = [MaxPooling2D(pool_size, 
                                               strides=1, 
                                               padding="same", 
                                               name=f'pool_{pool_size}x{pool_size}'
                                               ) for pool_size in self.pool_sizes 
                                ]
        
       
        self.pooling_concat = Concatenate(axis=-1, name='pool_concat')

        self.post_conv = tf.keras.Sequential(name='post_conv_block')

        for i in range(self.depth - 1):   # depth = 1 for yolov7_tiny
            post_id = i * 2 
            self.post_conv.add(Conv2D_BN(filters = self.hidden_channels,
                                        kernel_size =1,
                                        strides =1,
                                        bn_epsilon = self.bn_epsilon,
                                        bn_momentum = self.bn_momentum,
                                        activation = self.act,
                                        name = f"post_conv_{post_id+1}" )
                
            )

            self.post_conv.add(Conv2D_BN(filters = self.hidden_channels,
                                        kernel_size =3,
                                        strides =1,
                                        bn_epsilon = self.bn_epsilon,
                                        bn_momentum = self.bn_momentum,
                                        activation = self.act,
                                        name = f"post_conv_{post_id+2}" )
                
            )

        if self.depth == 1:  # For yolov7_tiny
            self.post_conv.add(Conv2D_BN(filters = self.hidden_channels,
                                        kernel_size =1,
                                        strides =1,
                                        bn_epsilon = self.bn_epsilon,
                                        bn_momentum = self.bn_momentum,
                                        activation = self.act,
                                        name = "post_conv_1" ) 
            )                 



        self.output_concat = Concatenate(axis=-1, name='output_concat')

        self.output_conv = Conv2D_BN(filters = self.out_channels,
                                kernel_size =1,
                                strides =1,
                                bn_epsilon = self.bn_epsilon,
                                bn_momentum = self.bn_momentum,
                                activation = self.act,
                                name = "output_conv" )
        
    def call(self, x):

        short = self.short_conv(x) 
        deep =  self.prev_conv(x) 

        if not self.SPPFCSPC :
            feats = [deep]+[self.MaxPooling2D_list[i](deep) for i in range(len(self.pool_sizes))]
  
        else:
            feats = [deep]
            for i in range(len(self.pool_sizes)):
                feat = self.MaxPooling2D_list[i](feats[-1])
                feats.append(feat)

        deep =  self.pooling_concat (feats) 
        deep = self.post_conv(deep)
        output = self.output_concat([deep, short])
        output = self.output_conv(output)
        return output
    
    def get_config(self):
        config = super(SPPCSPC, self).get_config()
        config.update(
                {
                "out_channels": self.out_channels,
                "expansion": self.expansion,
                "hidden_channels": self.hidden_channels,
                "SPPFCSPC": self.SPPFCSPC,
                "lite_type": self.lite_type,
                "depth": self.depth,
                "pool_sizes": self.pool_sizes,
                "bn_epsilon": self.bn_epsilon,
                "bn_momentum": self.bn_momentum,
                "act": self.act
                }
        )
        return config
    

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
class SPPF(tf.keras.layers.Layer):
    VESRION = '1.0.0'
    r"""SPPF_Layer(spatial_pyramid_pooling_fast)
    spatial_pyramid_pooling_fast, used in YoloV8
    https://blog.csdn.net/weixin_43694096/article/details/126354660
    """
    def __init__(self, 
                out_channels : int,
                pool_size : int =5,
                bn_epsilon : float= 1e-5,
                bn_momentum : float= 0.9,
                activation = 'silu',
                name='SPPF',
                **kwargs):
        
        super(SPPF, self).__init__(name=name)
        
        if not isinstance(pool_size, int) :
            raise TypeError(f"pool_sizes must be int type @{self.__class__.__name__}"
                            f"but got {type(pool_size)}"
            )
        if (not isinstance(bn_epsilon,(float))) or (not isinstance(bn_momentum,(float))):
            raise TypeError(f"bn_eps and  bn_momentum must be 'float' type @{self.__class__.__name__}"
                                f"but got eps:{type(bn_epsilon)}, momentum:{type(bn_momentum)}"
            )
        if not isinstance(activation,(str, type(None))):
            raise TypeError("activation must be 'str' type like 'relu'"
                         f"but got {type(activation)} @{self.__class__.__name__}"
            )

        self.out_channels = out_channels
        self.pool_size = pool_size
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.act = activation

    def build(self, input_shape):
        self.hidden_channels = int(input_shape[-1]//2)

        self.prev_conv = Conv2D_BN(filters = self.hidden_channels,
                                    kernel_size=1,
                                    strides=1,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum = self.bn_momentum,
                                    activation = self.act,
                                    name = 'prev_conv')
        
        self.post_conv = Conv2D_BN(filters = self.out_channels,
                                    kernel_size=1,
                                    strides=1,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum = self.bn_momentum,
                                    activation = self.act,
                                    name = 'post_conv')
        
        self.MaxPooling2D_list = [MaxPooling2D(self.pool_size, 
                                               strides=1, 
                                               padding="same", 
                                               name=f'pool_{self.pool_size}x{self.pool_size}_{i}'
                                               ) for i in range(3) 
                                ]
        
        self.pool_concat = Concatenate(axis=-1, name='pool_concat')

    def call(self, x):

        nn = self.prev_conv(x) 
        feats = [nn]
        for i in range(3):
            feat = self.MaxPooling2D_list[i](feats[-1])
            feats.append(feat)
        output = self.pool_concat(feats)
        output =  self.post_conv(output) 
        return output
    
    def get_config(self):
        config = super(SPPF, self).get_config()
        config.update(
                {
                "out_channels": self.out_channels,
                "hidden_channels": self.hidden_channels,
                "pool_size": self.pool_size,
                "bn_epsilon": self.bn_epsilon,
                "bn_momentum": self.bn_momentum,
                "act": self.act
                }
        )
        return config
    

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------

class SPP(tf.keras.layers.Layer):
    VESRION = '1.0.0'
    r"""SPP / SPPBottleneck (spatial pyramid pooling) used in YOLOv3-SPP / CSPNeXt
    spatial_pyramid_pooling,

    Architecture :
        in: (b,80,80,256) => CrossStagePartial_C3 => out: (b,80,80,256)
        SPP = {
                out_channels = 256,
                pool_sizes = [3,5,13] ,
        }
        -------------------------------------------------------------------------------------
        #[from, number, module, args]
        [-1, 1, Conv2D_BN, [128, 1, 1]],                 # (80,80,256)=>(80,80,128)    args ={filters, kernel_size, strides}
        [-1, 1, MP, [5, 1]],                             # (80,80,128) 
        [-2, 1, MP, [9, 1]],                             # (80,80,128) 
        [-3, 1, MP, [13, 1]],                            # (80,80,128) 
        [[-1, -2,-3,-4], 1, Concat, [-1]],               # (80,80,128)x4 => (80,80,512)
        [-1, 1, Conv2D_BN, [256, 1, 1]],                 # (80,80,512)=>(80,80,128) 

    References:
            - [Based on implementation of 'SPPBottleneck' @mmdet] (https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/csp_darknet.py#L67)
            - [More detail for SPP family] (https://blog.csdn.net/weixin_43694096/article/details/126354660)

    Args:
        out_channels (int) : The output channels of this Module.
        pool_sizes (int) :  Tuple/list of 3 integers representing the pooling sizes of MaxPooling2Ds, respectivly.
                defaults to [5,9,13]
        bn_epsilon (float) : epsilon of batch normalization , defaults to 1e-5.
        bn_momentum (float) : momentum of batch normalization, defaults to 0.9.
        activation (str) : activation used in DepthwiseConvModule and ConvModule, defaults to 'swish'.
        depoly (bool): determine depolyment config . default to None, 
                depoly = None => disable re-parameterize attribute (turn off switch_to_deploy), all cfgs depend on above args.
                depoly = True => to use deployment config, only conv layer will be bulit
                depoly = False => to use training config() , 
        name (str) : 'SPP'
    Note :


    Examples:
    ```python

        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        x = Input(shape=(256,256,128))
        out = SPP(out_channels=128,
                    pool_sizes = (5,9,13),
                    bn_epsilon= 1e-5,
                    bn_momentum = 0.9,
                    activation = 'silu',
                    deploy= False,
                    name='SPP')(x)

        model = Model(x, out)
        model.summary(100)
        print( model.get_layer('SPP').weights[0][:,:,0,0])
        for layer in model.layers:
            if hasattr(layer,'switch_to_deploy'):
                layer.switch_to_deploy()
                print('---------------------')
                for weight in layer.weights:
                    print(weight.name)
                tf.print(layer.get_config())   
        print( model.get_layer('SPP').weights[0][:,:,0,0])
        model.summary(100)     
    
    """
    def __init__(self, 
                out_channels : int,
                pool_sizes : Tuple[int] = (5,9,13),
                bn_epsilon : float= 1e-5,
                bn_momentum : float= 0.9,
                activation = 'swish',
                deploy : Optional[bool] = None,
                name='SPP',
                **kwargs):
        
        super(SPP, self).__init__(name=name)
        
        if not isinstance(pool_sizes,Sequence) or len(pool_sizes)!=3 and all(not isinstance(x,int) for x in pool_sizes):
            raise TypeError(f"bn_eps and  bn_momentum must be 'float' type @{self.__class__.__name__}"
                                f"but got eps:{type(bn_epsilon)}, momentum:{type(bn_momentum)}"
            )
        if (not isinstance(bn_epsilon,(float))) or (not isinstance(bn_momentum,(float))):
            raise TypeError(f"bn_eps and  bn_momentum must be 'float' type @{self.__class__.__name__}"
                                f"but got eps:{type(bn_epsilon)}, momentum:{type(bn_momentum)}"
            )
        if not isinstance(activation,(str, type(None))):
            raise TypeError("activation must be 'str' type like 'relu'"
                         f"but got {type(activation)} @{self.__class__.__name__}"
            )
        self.deploy = deploy
        self.out_channels = out_channels
        self.pool_sizes = pool_sizes
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.act_name = activation

    def build(self, input_shape):
        _,_,_,self.in_channels = input_shape

        if self.out_channels<0:
            self.out_channels = self.in_channels

        self.hidden_channels = int(self.in_channels//2)

        self.prev_conv = Conv2D_BN(filters = self.hidden_channels,
                                    kernel_size=1,
                                    strides=1,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum = self.bn_momentum,
                                    activation = self.act_name,
                                    deploy = self.deploy,
                                    name = 'prev_conv')
        
        self.post_conv = Conv2D_BN(filters = self.out_channels,
                                    kernel_size=1,
                                    strides=1,
                                    bn_epsilon = self.bn_epsilon,
                                    bn_momentum = self.bn_momentum,
                                    activation = self.act_name,
                                    deploy = self.deploy,
                                    name = 'post_conv')
        
        self.MaxPooling2D_list = [MaxPooling2D(pool_size, 
                                               strides=1, 
                                               padding="same", 
                                               name=f'pool_{pool_size}x{pool_size}'
                                               ) for pool_size in self.pool_sizes
                                ]
        
        self.pool_concat = Concatenate(axis=-1, name='pool_concat')

    def call(self, x):

        nn = self.prev_conv(x) 
        feats = [nn] + [MaxPooling2D(nn) for MaxPooling2D in self.MaxPooling2D_list]
        output = self.pool_concat(feats)
        output =  self.post_conv(output) 
        return output
    
    def get_config(self):
        config = super(SPP, self).get_config()
        config.update(
                {
                "reparam_deploy": self.out_channels,
                "out_channels": self.out_channels,
                "hidden_channels": self.hidden_channels,
                "pool_sizes": self.pool_sizes,
                "bn_epsilon": self.bn_epsilon,
                "bn_momentum": self.bn_momentum,
                "act": self.act_name,
                "params": super().count_params()
                }
        )
        return config
    
    def switch_to_deploy(self):

        if self.deploy or self.deploy==None:
            return
        'get fused weight of conv_bn_1 and remove its bn layer'
        self.prev_conv.switch_to_deploy()
        prev_conv_weights  = self.prev_conv.weights
        self.post_conv.switch_to_deploy()
        post_conv_weights  = self.post_conv.weights
        're build dw_conv_bn by seting input shape/ deploy = True / built = False'
        self.built = False
        self.deploy = True       
        super().__call__(tf.random.uniform(shape=(1,32,32,self.in_channels)))
        'update fused_weights to self.conv'
        self.prev_conv.set_weights(prev_conv_weights) 
        self.post_conv.set_weights(post_conv_weights) 