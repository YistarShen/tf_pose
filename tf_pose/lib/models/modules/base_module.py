#from keras.utils import tf_utils
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow import keras
from typing import  Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers

def get_shapes(tensors):
    """Gets shapes from tensors."""
    return tf.nest.map_structure(
        lambda x: x.shape if hasattr(x, "shape") else None, tensors
    )
class BaseModule:
    VERSION = "1.2.0"
    r""" BaseModule API
    flexible module class for tensorflow

    input transform ops support multi-stages with multi-features

    Args:
    
        name (str) :
        wrapped_model(str) : 
        scope_marks (int) : 
        bn_epsilon (float) : 
        activation ('swish') : 
        input_transforms (List[dict]) :
    Example:
        '''Python

        - multi_inputs-------------------------------------

        class mult_conv(BaseModule)
            def __init__(self, name=name, **kwargs )->None:
                super().__init__(name=name, **kwargs)
                
            def call(features):
                p3,p4, p5 = features  
                p3 = Conv2D_BN(filters = -1)(p3)
                p4 = Conv2D_BN(filters = -1)(p4)
                p5 = Conv2D_BN(filters = -1)(p5)
                return [p3,p4, p5]

        feat_p3 = Input((80,80,128))
        feat_p4 = Input((40,40,256))
        feat_p5 = Input((20,20,256))
        features = [feat_p3,feat_p4,feat_p5]
        out = mult_conv(features)
        model = Model(features, out)
        model.summary(200, expand_nested=True)

        - input transform for multi stages with multi-features-------------------------------------

        class ModuleTest(BaseModule):
            def __init__(self, 
                    activation : str ="relu", 
                    name='ModuleTest', 
                    **kwargs): 
                super().__init__(
                    name=name,  activation=activation,  **kwargs
                )    
            def call(self, inputs):
                print(inputs)
                return inputs
        
        ops_trans_xs1 = [ 
            [dict(type="Reshape", target_shape=(64,48,-1),name="stage1_Reshape1"), dict(type="Conv2D", filters=128, kernel_size=1,name="stage1_pwconv")], 
            [dict(type="Reshape", target_shape=(32,24,-1),name="stage1_Reshape2")]
        ]
        ops_trans_xs2 = [ 
            [dict(type="Reshape", target_shape=(64,48,-1),name="stage2_Reshape1"), dict(type="Conv2D", filters=128, kernel_size=1,name="stage2_pwconv")], 
            []
        ]
        xs_1 = [Input(shape=(64*48, 64)),Input(shape=(32*24, 128)) ]
        xs_2 = [Input(shape=(64*48, 64)),Input(shape=(32*24, 128)) ]
        x = [xs_1, xs_2]
        transforms = [ops_trans_xs1,ops_trans_xs2]

        out = ModuleTest(
            input_transforms = transforms,
            wrapped_model = False,
        )(x)
        model = Model(x,out)
        model.summary(200,expand_nested=False)
        model.output_shape

    """
    def __init__(self,
                name,
                bn_epsilon : float= 1e-5,
                bn_momentum : float= 0.9,
                activation : str = None,
                deploy : Optional[bool] = None,
                input_transforms : Optional[List[dict] ] = None, 
                scope_marks : str ="_",
                wrapped_model : bool = False)->None:
        
        self.built = False


        if (not isinstance(bn_epsilon,(float))) or (not isinstance(bn_momentum,(float))):
            raise TypeError(
                f"bn_eps and  bn_momentum must be 'float' type @{self.__class__.__name__}"
                f"but got eps:{type(bn_epsilon)}, momentum:{type(bn_momentum)}"
            )
        if not isinstance(activation,(str, type(None))):
            raise TypeError(
                "activation must be 'str' type like 'relu'"
                f"but got {type(activation)} @{self.__class__.__name__}"
            )
        
        
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.act_name = activation 
        self.deploy = deploy

        'optional CFG to wrap model'
        self.wrapped_model = wrapped_model
    

        self.name = f"{name}"+ scope_marks if not self.wrapped_model else name

        'optional CFG for input_transform'
        # if input_transforms is not None and input_transforms!=[]:
        #     self.input_transforms_map = self.parse_input_transforms(
        #         input_transforms, methods=[], scope=""
        #     )
        if input_transforms is not None and input_transforms!=[]:
            self.input_transforms_map = self.parse_input_transforms(
                input_transforms, methods=[], scope=self.name
            )
        #self.name = f"{name}"+ scope_marks if not self.wrapped_model else name

    def build(self,input_shape):

        """Creates the variables of the layer (for subclass implementers).

        This is a method that implementers of subclasses of `Layer` or `Model`
        can override if they need a state-creation step in-between
        layer instantiation and layer call. It is invoked automatically before
        the first execution of `call()`.

        This is typically used to create the weights of `Layer` 
        (at the discretion of the subclass implementer).

        Args:
          input_shape: Instance of `TensorShape`, or list of instances of
            `TensorShape` if the layer expects a list of inputs
            (one instance per input).
        """
        if self.built:
            return 
        self.built  = True
        
    def call(inputs) ->tf.Tensor:

        """This is where the module's logic lives.

        The `call()` method may not create state.  
        It is recommended to create state, including
        `tf.Variable` instances and nested `Layer` instances,
         in `__init__()`, or in the `build()` method that is
        called automatically before `call()` executes for the first time.

        Args:
          inputs: Input tensor, or dict/list/tuple of input tensors.
            The first positional `inputs` argument is subject to special rules:
            - `inputs` must be explicitly passed. A layer cannot have zero
              arguments, and `inputs` cannot be provided via the default value
              of a keyword argument.
            - NumPy array or Python scalar values in `inputs` get cast as
              tensors.
            - Keras mask metadata is only collected from `inputs`.
            - Layers are built (`build(input_shape)` method)
              using shape info from `inputs` only.
      

        Returns:
          A tensor or list/tuple of tensors.
         """ 
        return inputs
    
    def transforms_ops(self,inputs):
        """ 
        inputs : xs = [ [Input(shape=(64,48, 64)),Input(shape=(32,24, 128))], [Input(shape=(64,48, 64)),Input(shape=(32,24, 128))] ]
        trans : [ [seq_stage1_feat1, seq_stage1_feat2], [seq_stage2_feat1, seq_stage2_feat2]]
         
        """
        for ith_stage, (xs, transforms) in enumerate(zip(inputs,self.input_transforms_map)):
            for jth_feat in range(len(xs)):
                inputs[ith_stage][jth_feat] =  transforms[jth_feat](inputs[ith_stage][jth_feat]) 
        return inputs
    def data_adapter(self,inputs):
        #KerasTensor
        if not isinstance(inputs, list): 
            return inputs
        elif len(inputs) == 1:
            return inputs[0]
        else:
            return inputs
    
    def __call__(self, inputs)->tf.Tensor:
        
        inputs = self.data_adapter(inputs)

        if  not self.built :
            self.build(get_shapes(inputs))

            if self.wrapped_model :
      
                input_layers = tf.nest.map_structure(
                    lambda x: Input( x.shape[1:] ) if hasattr(x, "shape") else None, inputs
                )
                if hasattr(self,'input_transforms_map') :
                    outputs =self.call(  
                        self.inputs_transformer(
                            input_layers, self.input_transforms_map, []
                        ) 
                    )
                else:
                    outputs = self.call( input_layers )

                self.model = Model(
                    input_layers, outputs, name= self.name
                )
            else:
                if hasattr(self,'input_transforms_map'):
                    inputs = self.inputs_transformer(
                        inputs, self.input_transforms_map, []
                    ) 
            self.built  = True
        
        return self.model(inputs) if hasattr(self,'model') else self.call(inputs) 
 
    
    def get_config(self):
        config = {
            "wrapped_model" : self.wrapped_model,
            "name": self.name,
            "bn_epsilon": self.bn_epsilon,
            "bn_momentum": self.bn_momentum,
            "act": self.act_name,
        }
        return config
    
    
    def inputs_transformer(self,
                        xs: list,
                        transforms_map : list,
                        transformed_inputs=[]):
        
        if hasattr(xs,'shape') and isinstance(transforms_map, keras.Sequential):
            return transforms_map(xs)

        if all([ hasattr(x,'shape')for x in xs]):
            outputs = []
            for x, transform_seq in  zip(xs, transforms_map):
               outputs.append( transform_seq(x))
            return outputs
        
        for x, transform in zip(xs, transforms_map):
            stage_outputs = self.inputs_transformer(x, transform, [])
            transformed_inputs.append(stage_outputs)
        
        return transformed_inputs  
      
    def parse_input_transforms(
                    self,
                    cfg_list :list, 
                    methods : list = [], 
                    scope : str = "")->list:
        
        if not all([isinstance(x,(dict,list)) for x in cfg_list]):
            raise TypeError(
                "found no-support type of item in input_transforms, \n"
                "i.e. multi-input-stages format : [[dict,dict], [dict]] or [dict,dict]\n"
                "i.e. single-input-stages format : [dict,dict]\n"
                f"but got {cfg_list} @"
            )
        'parse cfg_list = [], no ops'
        if cfg_list == [] :
            input_transform_seq = tf.keras.Sequential(
                [layers.Layer(name='Identity')], name=scope+"Transforms"
            )
            return input_transform_seq
        
        'parse dict in a list'
        if all([isinstance(x,dict) for x in cfg_list]):
            input_transform_seq = tf.keras.Sequential(
                    name=scope+"Transforms"
            )
       
            for cfg in cfg_list.copy() :
                input_transform_seq.add( 
                    getattr(layers,cfg.pop("type"))(**cfg)
                ) 
            return input_transform_seq
        
        if 'stage' in scope and "feat" in scope:
            raise ValueError(
                "only support to multi stages with multi features"
                "multi features in a stage had sub features were invalid"
            ) 
        
        'list'
        for ith, sub_cfg_list in enumerate(cfg_list):
            if 'stage' in scope:
                seq_name = scope+f"_feat{ith+1}"
            else:
                seq_name = scope+f"_stage{ith+1}"   
            input_transform_seq = self.parse_input_transforms(
                    sub_cfg_list, [], seq_name
            )
            methods.append(input_transform_seq)
        return methods