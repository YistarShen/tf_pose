
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from lib.codecs import BaseCodec
from lib.models.builder import build_backbone
from lib.Registers import MODELS, CODECS
from lib.models.builder import build_backbone, build_loss, build_metric
from tensorflow import Tensor
import tensorflow as tf
import copy, warnings,os
type_opt = tf.keras.optimizers.Optimizer
type_exp_opt = tf.keras.optimizers.experimental.Optimizer
type_loss = tf.keras.losses.Loss
type_layer = tf.keras.layers.Layer

#-------------------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------------------
class BaseModel(tf.keras.Model):
    VERSION = '1.2.0'
    r"""BaseModel Task
    Author : Dr. David Shen
    Date : 2024/2/23
    This class defines the basic functions to build up model by. 
    Any class backbone that inherits this class should 


    Args:
        backbone  : 
        neck  :  
        head  : 
        extract_layres  :  
        codec : 
        optimizer : 
        losses  : 
        metrics : 
        pretrained_weights_path (bool): 

    Note (some useful  ):
        self._is_graph_network = False
        self.inputs = None
        self.outputs = None
        self.input_names = None
        self.output_names = None
    """
    def __init__(self,
        backbone : Union[dict, tf.keras.Model] ,
        neck : Optional[Union[dict, tf.Module, type_layer]] = None,
        head : Optional[Union[dict, tf.Module, type_layer]] = None,
        extract_layres : Optional[List[str]] = [],
        aux_extract_layres : Optional[List[str]] = [],
        aux_module :  Optional[Union[dict, tf.Module, type_layer]] = None,
        codec : Optional[dict]=None,
        optimizer : Optional[dict] = None,
        losses :  Optional[Union[list, dict]] = None,
        metrics : Optional[Union[list, dict]] = None,
        pretrained_weights_path : Optional[str] = None, 
        jit_compile  : bool= False,
        *args, **kwargs):
        #--------<1.>  Build Model by backbone, neck, head------
        backbone = self.parse_backbone(backbone)
        if neck is not None:
            neck = self.parse_modules(neck, module_type='neck')
        if head is not None:
            head = self.parse_modules(head, module_type='head')
        if aux_module is not None and aux_extract_layres!=[]:
            aux_module = self.parse_modules(aux_module, module_type='aux_branch')    
         
        super().__init__(
                inputs = backbone.input,
                outputs = self._modeling(
                    backbone, neck, head, extract_layres, aux_extract_layres, aux_module
                ),
                *args, **kwargs)
        
        self.extract_layres = extract_layres
        del backbone, neck, head
        #-------------------------------------------------------------------

       
        #--------<2.> compile model if losses and optimizer both are not None ------
        if losses is not None and optimizer is not None :
            self.compile(
                optimizer = optimizer,
                loss = losses,
                metrics = metrics,
                jit_compile = jit_compile
            )
        self.load_pretrained_weights(pretrained_weights_path)  

        #------------<2.> codec for encoder and ----------------------
        'BUILD CODEC for decoder and encoder (optional)'
        self.encoder = None
        self.decoder = None
        if codec is not None :
            self.parse_codec(codec)
        else:
            warnings.warn("codec is None, you may not gen targets used in training") 

        self._has_encoder = True if isinstance(self.encoder, Callable) else False
        self._has_decoder = True if isinstance(self.decoder, Callable) else False 


        #------------<3.> cfg for training ----------------------
        self.batch_size = None
        self.initialized_task = False
        self.data_format = None

    def parse_codec(self, codec): 
        if  isinstance(codec, dict) :
            self.codec = CODECS.build(copy.deepcopy(codec))
        elif isinstance(codec, BaseCodec):
            self.codec =  codec
        else:
            raise TypeError(
                f"codec_cfg must be 'dict' or 'BaseCodec' type, but got {type(codec)} @BaseModel "
            ) 
        
        if not (hasattr(self.codec ,'batch_encode') and hasattr(self.codec ,'batch_decode')):
            raise TypeError(
                f"codec must have 'batch_encode' and  'batch_decode' attributes"
            ) 
        
        #self.codec.embedded_codec = True

        if hasattr(self,'codec'):
            'model apply codec, so set embedded_codec=True, its default is False'
            self.codec.embedded_codec = True
            
            assert isinstance(self.codec.batch_encode, Callable), \
            "self.codec.batch_encode is not callable"
            #self.encoder = self.codec.batch_encode  
            self.encoder = lambda *args, **kwargs: self.codec(*args, **kwargs, codec_type='encode')

            assert isinstance(self.codec.batch_decode, Callable), \
            "self.codec.batch_decode is not callable"
            #self.decoder = self.codec.batch_decode    
            self.decoder = lambda *args, **kwargs: self.codec(*args, **kwargs, codec_type='decode')


        
  
    def load_pretrained_weights(self, pretrained_weights_path):
        if pretrained_weights_path is not None:
           if isinstance(pretrained_weights_path, str) and os.path.exists(pretrained_weights_path) :
                super().load_weights(
                    pretrained_weights_path, by_name=True, skip_mismatch=False
                )
                print(
                    f"already load weights from {pretrained_weights_path} with [ by_name=True, skip_mismatch=False ]"
                )
           else: 
                warnings.warn(
                    "Warning!!!!!!! Cannot find valid pretrained weights path \n" 
                    f"in {pretrained_weights_path}  @{self.__class__.__name__}"
                )    
  
    def parse_backbone(self,backbone):
        if backbone is None:
            raise ValueError(
                f"backbone is None type @{self.__class__.__name__}"
            )
        if isinstance(backbone, dict):
            return build_backbone(backbone)
        elif isinstance(backbone, tf.keras.Model):
            return backbone
        else:
            raise TypeError(
                "backbone must be 'dict' or 'tf.keras.Model'"
                f"but got {type(backbone)} @{self.__class__.__name__}"
            )
        
    def parse_modules(self,
                    module, 
                    module_type='neck'):

        if isinstance(module, dict):
            return MODELS.build(module)
        elif isinstance(module, Callable):
            return module
        else:
            raise TypeError(
                f"{module_type}'s type must be callable function"
                f"but got {type(module)}  @{self.__class__.__name__}"
            )
        
    def _modeling(
            self, 
            backbone, 
            neck,
            head,
            extract_layers_list : List[str]=[],
            aux_extract_layres : Optional[List[str]] = [],
            aux_module :  Optional[Union[dict, tf.Module, type_layer]] = None,
        ) -> Union[Tuple[Tensor], Tensor]:
        """Extract features.
        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).
        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """

        assert isinstance(extract_layers_list, List), \
        "extract_layres must be a List, i.e [] or ['conv_p3']"

        if extract_layers_list == []:
            #x = backbone.output #modify to output for multi-stages with multi-feats
            ''' 
            if backbone.output is multi-feats  , the type of x is list 
            if backbone.output is single feat , the type of keras Tensor
            i.e. 
                x : KerasTensor
                x : [KerasTensor]
                x : [KerasTensor, KerasTensor]
            '''
            x = backbone.output
        else:
            x = []
            for extract_layer in extract_layers_list :
                if isinstance(extract_layer, str) : 
                    print("extract_layer : ", extract_layer)
                    x.append(backbone.get_layer(extract_layer).output)
                else:
                    raise RuntimeError(
                        "extract_layer(name) must be 'str"
                    )   
                 
        from keras.engine.keras_tensor import KerasTensor       
        if neck is not None:
            if isinstance(x, KerasTensor):
                x = neck(x)
            elif isinstance(x,list):
                x = neck(x) if len(x)!=1 else neck(*x)
            else:
                raise RuntimeError(
                    " input type of neck must be  'KerasTensor' or List[KerasTensor]"
                    f"but got {type(x)} @{self.__class__.__name__}"
                )   

        if head is not None:
            if isinstance(x, KerasTensor):
                x = head(x)
            elif isinstance(x,list):
                x = head(x) if len(x)!=1 else head(*x)
            else:
                raise RuntimeError(
                    " input type of head must be  'KerasTensor' or 'List[KerasTensor]' "
                     f"but got {type(x)} @{self.__class__.__name__}"
                )   
            
        if aux_module is not None  and aux_extract_layres!=[]:
            if not all([ isinstance(layer_name, str) for layer_name in tf.nest.flatten(aux_extract_layres)] ):
                raise TypeError(
                    "all items in aux_extract_layres list must be 'str' type "
                    f"but got {aux_extract_layres} @{self.__class__.__name__}"
                ) 
            aux_x = tf.nest.map_structure(
                lambda x : backbone.get_layer(x).output, aux_extract_layres
            )
            aux_x = aux_module(aux_x) 
            return tf.nest.flatten([aux_x,  x])
        
        return x        
        
    # def extract_feat(
    #         self, 
    #         backbone, 
    #         neck,
    #         head,
    #         extract_layers_list : List[str]=[]) -> Union[Tuple[Tensor], Tensor]:
    #     """Extract features.
    #     Args:
    #         inputs (Tensor): Image tensor with shape (N, C, H ,W).
    #     Returns:
    #         tuple[Tensor]: Multi-level features that may have various
    #         resolutions.
    #     """

    #     assert isinstance(extract_layers_list, List), \
    #     "extract_layres must be a List, i.e [] or ['conv_p3']"

    #     if extract_layers_list == []:
    #         #x = backbone.output #modify to output for multi-stages with multi-feats
    #         ''' 
    #         if backbone.output is multi-feats  , the type of x is list 
    #         if backbone.output is single feat , the type of keras Tensor
    #         i.e. 
    #             x : KerasTensor
    #             x : [KerasTensor]
    #             x : [KerasTensor, KerasTensor]
    #         '''
    #         x = backbone.output
    #     else:
    #         x = []
    #         for extract_layer in extract_layers_list :
    #             if isinstance(extract_layer, str) : 
    #                 print("extract_layer : ", extract_layer)
    #                 x.append(backbone.get_layer(extract_layer).output)
    #             else:
    #                 raise RuntimeError(
    #                     "extract_layer(name) must be 'str"
    #                 )   
                 
    #     from keras.engine.keras_tensor import KerasTensor       
    #     if neck is not None:
    #         if isinstance(x, KerasTensor):
    #             x = neck(x)
    #         elif isinstance(x,list):
    #             x = neck(x) if len(x)!=1 else neck(*x)
    #         else:
    #             raise RuntimeError(
    #                 " input type of neck must be  'KerasTensor' or List[KerasTensor]"
    #                 f"but got {type(x)} @{self.__class__.__name__}"
    #             )   
            
    #         # if not isinstance(x,list) or (isinstance(x,list) and len(x)!=1):
    #         #     x = neck(x)
    #         # else:
    #         #     'x is list type with len=1'
    #         #     x = neck(*x)

    #         # if len(x)==1:
    #         #     x = neck(*x)
    #         # else:
    #         #     x = neck(x)

    #     if head is not None:
    #         if isinstance(x, KerasTensor):
    #             x = head(x)
    #         elif isinstance(x,list):
    #             x = head(x) if len(x)!=1 else head(*x)
    #         else:
    #             raise RuntimeError(
    #                 " input type of head must be  'KerasTensor' or 'List[KerasTensor]' "
    #                  f"but got {type(x)} @{self.__class__.__name__}"
    #             )   
            
    
    #         # if not isinstance(x,list) or (isinstance(x,list) and len(x)!=1):
    #         #     x = head(x)
    #         # else:
    #         #     'x is list type with len=1'
    #         #     x = head(*x)

   
    #         # if len(x)==1:
    #         #     x = head(*x)
    #         # else:
    #         #     x = head(x)
    #     return x

    @classmethod
    def set_optimizer(cls,
                    optimizer_cfg : Optional[Union[dict, str, type_opt, type_exp_opt]]=None) : 
        
        # if isinstance(optimizer_cfg, type_exp_opt) or isinstance(optimizer_cfg, type_opt): 
        #     return optimizer_cfg

        if isinstance(optimizer_cfg, (type_exp_opt,type_opt,str)) : 
            return optimizer_cfg    
        
        if isinstance(optimizer_cfg,dict): 
            if optimizer_cfg.get('type',None) is None :
                raise ValueError(
                    "if optimizer_cfg is dict type, it must have 'type' key i.e. {type='adam'}"
                    f"but got {optimizer_cfg}"
                )  
            opt_cfg = optimizer_cfg.copy()
            if opt_cfg['type'].startswith('experimental.'):
                opt = getattr(
                    tf.keras.optimizers.experimental, 
                    opt_cfg.pop('type')[len('experimental.'):] 
                )(**opt_cfg)
            else:
                opt = getattr(
                    tf.keras.optimizers, opt_cfg.pop('type') 
                )(**opt_cfg)
        else:
            raise TypeError(
                "if optimizer_cfg is not dict type ,"
                "it must be the type of {type_exp_opt} or {type_opt} or str"
                f"but got {type(optimizer_cfg)}"
            )
            #opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        return opt

    def build_loss_fun(self,
                    losses_cfg : Union[List[dict],Dict[str,dict],str]) -> Dict[str, tf.losses.Loss] : 
        
        
        if isinstance(losses_cfg, (dict,list)):
            assert len(losses_cfg) == len(self.outputs), \
            f'loss_fn num is {len(losses_cfg)},but num of model_outputs is {len(self.outputs)} @build_loss_fun'
        
        losses_dict = dict()
        if isinstance(losses_cfg, dict):
            'dict needs mapping to output.names'
            if not all(isinstance(x,(dict,str,tf.losses.Loss)) for x in losses_cfg.values()):
                raise TypeError(
                    "type of item in losses_cfg(dict) is invalid" 
                    "losses_cfg must be [Dict | Str | tf.keras.losses]"
                )  

            losses_dict = losses_cfg.copy()
            for (k,v), output_layer_name in zip(losses_cfg.items(),self.output_names):
                assert k == output_layer_name, \
                f"losses_cfg key : {k}  doesn't match  model_output_name : {output_layer_name}"
                losses_dict[k] =  build_loss(v) if isinstance(v, dict) else v

        elif isinstance(losses_cfg, list):
            'list : [str, tf.losses.Loss, dict]'
            if not all(isinstance(x,(dict,str,tf.losses.Loss)) for x in losses_cfg):
                raise TypeError(
                    "type of item in losses_cfg(List) is invalid" 
                    "losses_cfg must be [Dict | Str | tf.keras.losses]"
                )            
            for k,v in zip(self.output_names, losses_cfg) :
                losses_dict[k] = build_loss(v) if isinstance(v, dict) else v
        else:
            raise RuntimeError(
                "type of losses_cfg is invalid, losses_cfg must be ( List | Dict )"
                f"but got {type(losses_cfg)} @build_loss_fun in {self.__class__.__name__}"
            )   
        return losses_dict 



    def build_metric_fun(self,
                        metrics_cfg : Optional[Union[List,Dict]]=None) -> Union[Dict[str, tf.keras.metrics.Metric], None] : 
        
        if metrics_cfg == None:
            return None
        
        if not isinstance(metrics_cfg, (dict,list)):
            raise TypeError(
                "type of metrics_cfg is invalid, metrics_cfg must be  List or Dict "
                f"but got {type(metrics_cfg)}"
            )  

        metrics_dict = dict()
        if isinstance(metrics_cfg, list):
            'list'

            if not all(isinstance(x,(dict,str,tf.keras.metrics.Metric)) for x in metrics_cfg):
                raise TypeError(
                    "type of item in metrics_cfg(List) is invalid" 
                    "all items in metrics_cfg(List) must be type[Dict | Str | tf.keras.losses] "
                ) 
            
            # if len(metrics_cfg)==1:
            #     v = metrics_cfg[0]
            #     metrics_dict[self.output_names[-1]] = build_metric(v) if isinstance(v, dict) else v
            # elif len(metrics_cfg) != len(self.outputs) :
            #     raise ValueError(
            #         f"num of metric_fn  is {len(metrics_cfg)}"
            #         f"but num of model_outputs is {len(self.outputs)}"
            #     ) 
            

            if len(metrics_cfg) != len(self.outputs) :
                if len(metrics_cfg)==1:
                    v = metrics_cfg[0]
                    metrics_dict[self.output_names[-1]] = build_metric(v) if isinstance(v, dict) else v
                else:
                    raise ValueError(
                        f"num of metric_fn  is {len(metrics_cfg)}"
                        f"but num of model_outputs is {len(self.outputs)}"
                    ) 
            else:
                for k,v in zip(self.output_names, metrics_cfg) :
                    metrics_dict[k] = build_metric(v) if isinstance(v, dict) else v

        elif isinstance(metrics_cfg, dict):
            'dict'
            if not all(isinstance(x,(dict,str,tf.keras.metrics.Metric)) for x in metrics_cfg.values()):
                raise TypeError(
                    "type of item in metrics_cfg(dict) is invalid" 
                    "all items in metrics_cfg must be [Dict | Str | tf.keras.metrics.Metric]"
                )  
            metrics_dict = metrics_cfg.copy()
            for k,v in metrics_cfg.items():
                if k not in self.output_names:
                    raise ValueError(
                        f"metrics_cfg.key : {k}  must be in {self.output_names}"
                        "if metrics_cfg is a dict "
                    )   
                metrics_dict[k] = build_metric(v) 
        else:
            'other type'
            metrics_dict = None
    
        return metrics_dict  


    def compile(self, 
                optimizer :dict,     
                loss : Union[List[dict],Dict[str,dict]], 
                metrics : Optional[Union[List[dict],Dict[str,dict]]]=None,
                jit_compile : bool = False,
                **kwargs):
        r"""
        
        tf_model.compile(
                    optimizer='rmsprop',
                    loss=None,
                    metrics=None,
                    loss_weights=None,
                    weighted_metrics=None,
                    run_eagerly=None,
                    steps_per_execution=None,
                    jit_compile=None,
                    **kwargs
                )
        """
        opt = self.set_optimizer(optimizer) 
        losses = self.build_loss_fun(loss) 
        metrics = self.build_metric_fun(metrics)

        if jit_compile:
            warnings.warn(
                f"warning : 'BaseModel' will compile the tf-model with XLA........@{self.__class__.__name__}"
                "XLA is an optimizing compiler for machine learning, \n "
                "however jit_compile=True may not necessarily work for all models."
            ) 
  
        super().compile(
            optimizer = opt, 
            loss = losses, 
            metrics = metrics, 
            jit_compile = jit_compile,
            **kwargs
        )
        print(
            f"already compile model with optimizer and loss !!!!! @{self.__class__.__name__}"
        )

    def compute_loss(self, x, y, y_pred, sample_weight, **kwargs):

        if self.data_format==tuple:
            self.y_true = y
            self.y_pred = y_pred
            self.sample_weight = sample_weight
            del sample_weight
            del y
            del y_pred
            return super().compute_loss(
                    x=x, 
                    y= self.y_true, 
                    y_pred=self.y_pred,
                    sample_weight=self.sample_weight
            )
        
        if not self._has_encoder:
            raise ValueError(
                "need to build encoder before computing loss \n"
                f"if sample is dict not tuple @{self.__class__.__name__}"
            )
        
        y = self.encoder(
            y, y_pred = y_pred if self.codec.ENCODER_USE_PRED else None
        )
       
        if y.get('y_true',None) is None :
            raise ValueError(
                "cannot find required 'y_true' key in data from encoder \n"
                f"@{self.__class__.__name__} "
        )

        self.y_true = y.pop('y_true')
        self.sample_weight = y.pop('sample_weight', None)
        self.y_pred = y_pred if y.get('y_pred',None) is None else y.pop('y_pred') 
        del y_pred
        del y

        return  super().compute_loss(
            x=x, 
            y=self.y_true, 
            y_pred=self.y_pred, 
            sample_weight=self.sample_weight
        )    
    
    # def compute_loss(self, x, y, y_pred, sample_weight, **kwargs):

    #     #self.y_true = tf.Variable(tf.zeros(shape=(batch_size, *self.output_shape[1:])), dtype=tf.float32)
    #     #self.sample_weight = tf.Variable(tf.zeros(shape=(4,17,3)), dtype=tf.float32)
    
    #     if self.data_format==tuple:
    #         self.y_true = y
    #         self.sample_weight = sample_weight
    #         del sample_weight
    #         del y
    #         return super().compute_loss(
    #                 x=x, 
    #                 y= self.y_true, 
    #                 y_pred=y_pred,
    #                 sample_weight=self.sample_weight
    #         )
        
    #     if not self._has_encoder:
    #         raise ValueError(
    #             "need to build encoder before computing loss \n"
    #             f"if sample is dict not tuple @{self.__class__.__name__}"
    #         )
        
    #     y = self.encoder(
    #         y, y_pred if self.codec.ENCODER_USE_PRED else None
    #     )

    #     if y.get('y_true',None) is None :
    #         raise ValueError(
    #             "cannot find required 'y_true' key in data from encoder \n"
    #             f"@{self.__class__.__name__} "
    #     )

    #     self.y_true = y.pop('y_true')
    #     #self.sample_weight = y.get('sample_weight', None)
    #     self.sample_weight = y.pop('sample_weight', None)
    #     del y
    #     return  super().compute_loss(
    #         x=x, 
    #         y=self.y_true, 
    #         y_pred=y_pred, 
    #         sample_weight=self.sample_weight
    #     )    
    

    
    def compute_metrics(self, x, y, y_pred, sample_weight):
        del y # we use updated self.y_true instead of y_true
        return super().compute_metrics(
            x, self.y_true, self.y_pred, self.sample_weight
        )
    
    def train_step(self, data):
        """The logic for one training step.
        def train_step(self, data):
            x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
            # Run forward pass.
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compute_loss(x, y, y_pred, sample_weight)
            self._validate_target_and_loss(y, loss)
            # Run backwards pass.
            self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
            return self.compute_metrics(x, y, y_pred, sample_weight)
        """
        #y = {k:v for k,v in data.items()}
        if self.data_format ==dict:
            x = data.pop('image')
            return super().train_step((x, data))
        return super().train_step(data) 
    
    def test_step(self,data):
        if self.data_format==dict:
            x = data.pop('image')
            return super().test_step((x, data))
        return super().test_step(data) 

    def init_task(self,*arg,**kwargs):
        
        if self.initialized_task:
            return 
        
        print("init ModelingTask ..............")

        'get one iterative sample from training_data (tf.data.Dataset) '
        data =  kwargs['x'] if 'x' in kwargs else arg[0]
        if isinstance(data,tf.data.Dataset) :
            samples = next(iter(data))
            if not isinstance(samples,dict) and not isinstance(samples,tuple):
                raise TypeError(
                    f"data type of given iterative sample must be dict[str,Tensor] or tuple[Tensor] " 
                    f"but got {type(samples)}@{self.__class__.__name__}"
                )
     
            if isinstance(samples,dict) and not self._has_encoder:
                raise ValueError(
                    "need to build encoder before computing loss \n"
                    f"if sample is dict not tuple @{self.__class__.__name__}"
                )         
        else:
            raise TypeError(
                "training_data (x) must be 'tf.data.Dataset' "
                f"but got {type(data)} @{self.__class__.__name__}"
            ) 
        self.data_format  = type(samples)
        del samples

        'tf.data.Dataset : validation_data'
        if 'validation_data' in kwargs and not isinstance(kwargs['validation_data'],tf.data.Dataset):
            raise TypeError(
                f"validation_data must be 'tf.data.Dataset' "
                f"but got {type(data)}@{self.__class__.__name__} "
            ) 
        self.initialized_task = True
        
    
    def fit(self,*arg,**kwargs):
        if not self.initialized_task :
            self.init_task(*arg,**kwargs)
        super().fit(*arg,**kwargs)
    
    def predict_step(self, *args):
        'only use Tensor list[Tensor] or Tuple[Tensor]'
        y_pred = super().predict_step(*args)
        if self._has_decoder :
            return y_pred, self.decoder(y_pred)
        return y_pred, None
    
    def predict(self, *arg,**kwargs):

        if 'x' in kwargs.keys() :
            data = kwargs.pop('x')
        else:
            data = arg[0]
            arg = arg[1:]
        
        y_pred, decode_pred = super().predict(
            x=data['image'] if isinstance(data, dict) else data, 
            *arg,
            **kwargs
        )
        if isinstance(data, dict):
            data['y_pred'] = y_pred
            data['decode_pred'] = decode_pred
            return data
        
        return decode_pred if decode_pred is not None else y_pred
       
      
        
    def vis_evaluate(
            self, batched_samples_list : list, vis_fn : Callable = None, **kwargs
    ) :
        # if not isinstance(data, tf.data.Dataset):
        #     raise TypeError(
        #         f"Currently, only support  data's type is tf.data.Dataset"
        #         f"but got {type(data)}@{self.__class__.__name__} "
        #     ) 
        if not isinstance(vis_fn, Callable):
            raise TypeError(
                f"vis_fn must be callable function"
                f"but got {type(vis_fn)}@{self.__class__.__name__} "
            ) 
        
        if self._has_encoder :
            batched_samples_list = [self.encoder(batched_samples) for batched_samples in batched_samples_list]

        'prediction'
        y_preds_list = []
        decoded_preds_list= []
        for batched_samples in batched_samples_list :
            y_preds = self.predict(batched_samples['image'])
            if self._has_decoder :
                y_preds , decoded_preds = y_preds
                decoded_preds_list.append(decoded_preds)
            y_preds_list.append(y_preds)


        # preds_list  = [self.predict(batched_samples['image']) for batched_samples in batched_samples_list] 

        # if self._has_decoder and type(preds_list[0])==tuple:
        #    y_preds_list= [ preds[0] for preds in preds_list] 
        #    decoded_preds_list= [ preds[1] for preds in preds_list] 
        # else:
        #     y_preds_list = preds_list

        vis_fn(
            batched_samples = batched_samples_list,
            y_preds = y_preds_list,
            decoded_preds = decoded_preds_list,
            **kwargs
        )

        

    
    # def predict_step(self,*args):
    #     'args is dict'
    #     if len(args)==1 and type(args[0])==dict:
    #         data = args[0]
    #         y_pred = super().predict_step(data['image'])
    #         data['y_pred'] =  y_pred
    #         if self._has_decoder :
    #             decoded_pred = self.decoder(y_pred)
    #             data['decoded_pred'] =  decoded_pred
    #         return data
    #     'general case'
    #     y_pred = super().predict_step(*args)
    #     if self._has_decoder :
    #         return self.decoder(y_pred)
     
    
    


    # def codec_test(self,data):

    #     if isinstance(data,dict) and self._has_encoder:
    #         y = self.encoder(
    #             data, 
    #             self(data['image']) if self.codec.ENCODER_USE_PRED else None
    #         )
    #         return y
    #     else:
    #         raise ValueError(
    #                 "need to build encoder before computing loss \n"
    #                 f"if sample is dict not tuple @{self.__class__.__name__}"
    #         )   

# class BaseModel(ABC, tf.keras.Model):
#     VERSION = '1.0.0'
#     r"""BaseModel
#     This class defines the basic functions to build up model by. 
#     Any class backbone that inherits this class should 

#     Note (some useful  ):
#         self._is_graph_network = False
#         self.inputs = None
#         self.outputs = None
#         self.input_names = None
#         self.output_names = None
#     """
#     def __init__(self,
#         backbone : Union[dict, tf.keras.Model] ,
#         neck : Optional[Union[dict, tf.Module, tf.keras.layers.Layer]] = None,
#         head : Optional[Union[dict, tf.Module, tf.keras.layers.Layer]] = None,
#         extract_layres : Optional[List[str]] = [],
#         codec : Optional[dict]=None,
#         optimizer : Optional[dict] = None,
#         losses :  Optional[Union[list, dict]] = None,
#         metrics : Optional[Union[list, dict]] = None,
#         pretrained_weights_path : Optional[str] = None, 
#         *args, **kwargs):

#         backbone = self.parse_backbone(backbone)
#         if neck is not None:
#             neck = self.parse_modules(neck, module_type='neck')
#         if head is not None:
#             head = self.parse_modules(head, module_type='head')
        
         
#         super().__init__(inputs = backbone.input,
#                         outputs = self.extract_feat(backbone ,
#                                                     neck, 
#                                                     head, 
#                                                     extract_layres),
#                         *args, **kwargs)
        
#         self.extract_layres = extract_layres
#         self.build(backbone.input_shape)

#         self.backbone = backbone
#         del backbone, neck, head
#         #-------------------------------------------------------------------

       
#         #--------<2.> compile model if losses and optimizer both are not None ------
#         if losses is not None and optimizer is not None :
#             self.compile(optimizer = optimizer,
#                         loss = losses,
#                         metrics = metrics)
        
#         self.load_pretrained_weights(pretrained_weights_path)  

#         #------------<2.> codec for encoder and ----------------------
#         'BUILD CODEC for decoder and encoder (optional)'
#         self.encoder = None
#         self.decoder = None
#         if codec is not None :
#             self.parse_codec(codec)
#         else:
#             warnings.warn("codec is None, you may not gen targets used in training") 

#     def load_pretrained_weights(self, pretrained_weights_path):
#         if pretrained_weights_path is not None:
#            if isinstance(pretrained_weights_path, str) and os.path.exists(pretrained_weights_path) :
#                 super().load_weights(pretrained_weights_path, 
#                                     by_name=True, 
#                                     skip_mismatch=False
#                 )
#                 print(f"already load weights from {pretrained_weights_path} with [ by_name=True, skip_mismatch=False ]")
#            else: 
#                 warnings.warn("Warning!!!!!!! Cannot find valid pretrained weights path \n" 
#                             f"in {pretrained_weights_path}  @{self.__class__.__name__}"
#                 )    
  
    
#     def parse_backbone(self,backbone):
#         if backbone is None:
#             raise ValueError(f"backbone is None type @{self.__class__.__name__}")
#         if isinstance(backbone, dict):
#             return build_backbone(backbone)
#         elif isinstance(backbone, tf.keras.Model):
#             return backbone
#         else:
#             raise TypeError(
#                 "backbone must be 'dict' or 'tf.keras.Model'"
#                 f"but got {type(backbone)} @{self.__class__.name}"
#             )
        
#     def parse_modules(self,
#                     module, 
#                     module_type='neck'):

#         if isinstance(module, dict):
#             return MODELS.build(module)
#         elif isinstance(module, Callable):
#             return module
#         else:
#             raise TypeError(f"{module_type}'s type must be callable function"
#                             f"but got {type(module)}  @{self.__class__.name}"
#             )
        
#     def parse_codec(self, codec): 
#         if  isinstance(codec, dict) :
#             #self.codec = KEYPOINT_CODECS.build(copy.deepcopy(codec))
#             self.codec = CODECS.build(copy.deepcopy(codec))
#         elif hasattr(codec,'batch_encode') and hasattr(codec,'batch_decode'):
#             self.codec = codec 
#         else:
#             raise TypeError(f"codec_cfg must be 'dict' type, but got {type(codec)} @pose_dataloader "
#                             "pose dataloader need to build completed codec including encoder and decoder") 
        
#         if hasattr(self,'codec'):
#             assert isinstance(self.codec.batch_encode, Callable), \
#             "self.codec.batch_encode is not callable"
#             self.encoder = self.codec.batch_encode  
#             assert isinstance(self.codec.batch_decode, Callable), \
#             "self.codec.batch_decode is not callable"
#             self.decoder = self.codec.batch_decode     
        
#     def extract_feat(self, 
#                     backbone, 
#                     neck,
#                     head,
#                     extract_layers_list : List[str]=[]) -> Union[Tuple[Tensor], Tensor]:
#         """Extract features.
#         Args:
#             inputs (Tensor): Image tensor with shape (N, C, H ,W).
#         Returns:
#             tuple[Tensor]: Multi-level features that may have various
#             resolutions.
#         """

#         assert isinstance(extract_layers_list, List), \
#         "extract_layres must be a List, i.e [] or ['conv_p3']"

#         if extract_layers_list == []:
#             x = backbone.output #modify to output for multi-stages with multi-feats
#             #x = backbone.outputs
#         else:
#             x = []
#             for extract_layer in extract_layers_list :
#                 if isinstance(extract_layer, str) : 
#                     print("extract_layer : ", extract_layer)
#                     x.append(backbone.get_layer(extract_layer).output)
#                 else:
#                     raise RuntimeError("extract_layer(name) must be 'str")    
               
#         if neck is not None:
#             if len(x)==1:
#                 x = neck(*x)
#             else:
#                 x = neck(x)

#         if head is not None:
#             if len(x)==1:
#                 x = head(*x)
#             else:
#                 x = head(x)
#         return x


#     @classmethod
#     def set_optimizer(cls,
#                     optimizer_cfg : Optional[Union[dict, type_opt, type_exp_opt]]=None) : 
        
#         if isinstance(optimizer_cfg, type_exp_opt) or isinstance(optimizer_cfg, type_opt): 
#             return optimizer_cfg
        
#         if isinstance(optimizer_cfg,dict): 
#             if optimizer_cfg.get('type',None) is None :
#                 raise ValueError("if optimizer_cfg is dict type, it must have 'type' key i.e. {type='adam'}"
#                                  f"but got {optimizer_cfg}"
#             )  
#             opt_cfg = optimizer_cfg.copy()
#             if opt_cfg['type'].startswith('experimental.'):
#                 opt = getattr(tf.keras.optimizers.experimental, 
#                             opt_cfg.pop('type')[len('experimental.'):] )(**opt_cfg)
#             else:
#                 opt = getattr(tf.keras.optimizers, 
#                             opt_cfg.pop('type') )(**opt_cfg)
#         else:

#             raise TypeError("if optimizer_cfg is not dict type ,"
#                         "it must be the type of {type_exp_opt} or {type_opt} "
#                         f"but got {type(optimizer_cfg)}"
#             )
#             #opt = tf.keras.optimizers.Adam(learning_rate=0.001)
#         return opt

#     def build_loss_fun(self,
#                     losses_cfg : Union[List[dict],Dict[str,dict]]) -> Dict[str, tf.losses.Loss] : 
        
        
#         if isinstance(losses_cfg, (dict,list)):
#             assert len(losses_cfg) == len(self.outputs), \
#             f'loss_fn num is {len(losses_cfg)},but num of model_outputs is {len(self.outputs)} @build_loss_fun'
        
#         losses_dict = dict()
#         if isinstance(losses_cfg, dict):
#             'dict needs mapping to output.names'
#             if not all(isinstance(x,(dict,str,tf.losses.Loss)) for x in losses_cfg.values()):
#                 raise TypeError("type of item in losses_cfg(dict) is invalid" 
#                                 "losses_cfg must be [Dict | Str | tf.keras.losses]")  

#             losses_dict = losses_cfg.copy()
#             for (k,v), output_layer_name in zip(losses_cfg.items(),self.output_names):
#                 assert k == output_layer_name, \
#                 f"losses_cfg key : {k}  doesn't match  model_output_name : {output_layer_name}"
#                 losses_dict[k] =  build_loss(v) if isinstance(v, dict) else v

#         elif isinstance(losses_cfg, list):
#             'list : [str, tf.losses.Loss, dict]'
#             if not all(isinstance(x,(dict,str,tf.losses.Loss)) for x in losses_cfg):
#                 raise TypeError("type of item in losses_cfg(List) is invalid" 
#                                 "losses_cfg must be [Dict | Str | tf.keras.losses]")            
#             for k,v in zip(self.output_names, losses_cfg) :
#                 losses_dict[k] = build_loss(v) if isinstance(v, dict) else v
#         else:
#             raise RuntimeError("type of losses_cfg is invalid"
#                                "losses_cfg must be ( List | Dict )"
#                                f"but got {type(losses_cfg)}")   
#         return losses_dict 



#     def build_metric_fun(self,
#                         metrics_cfg : Optional[Union[List,Dict]]=None) -> Union[Dict[str, tf.keras.metrics.Metric], None] : 
        
#         if metrics_cfg == None:
#             return None
        
#         if not isinstance(metrics_cfg, (dict,list)):
#             raise TypeError("type of metrics_cfg is invalid"
#                             "metrics_cfg must be  List or Dict "
#                             f"but got {type(metrics_cfg)}"
#         )  

#         metrics_dict = dict()
#         if isinstance(metrics_cfg, list):
#             'list'
#             if len(metrics_cfg) != len(self.outputs):
#                 raise ValueError(f"loss_fn num is {len(metrics_cfg)}"
#                                   f"but num of model_outputs is {len(self.outputs)}"
#             ) 
#             if not all(isinstance(x,(dict,str,tf.keras.metrics.Metric)) for x in metrics_cfg):
#                 raise TypeError("type of item in metrics_cfg(List) is invalid" 
#                                 "all items in metrics_cfg(List) must be type[Dict | Str | tf.keras.losses] "
#             )    
#             for k,v in zip(self.output_names, metrics_cfg) :
#                 metrics_dict[k] = build_metric(v) if isinstance(v, dict) else v

#         elif isinstance(metrics_cfg, dict):
#             'dict'
#             if not all(isinstance(x,(dict,str,tf.keras.metrics.Metric)) for x in metrics_cfg.values()):
#                 raise TypeError("type of item in metrics_cfg(dict) is invalid" 
#                                 "all items in metrics_cfg must be [Dict | Str | tf.keras.metrics.Metric]")  
#             metrics_dict = metrics_cfg.copy()
#             for k,v in metrics_cfg.items():
#                 if k not in self.output_names:
#                     raise ValueError(f"metrics_cfg.key : {k}  must be in {self.output_names}"
#                                       "if metrics_cfg is a dict "
#                 )   
#                 metrics_dict[k] = build_metric(v) 
#         else:
#             'other type'
#             metrics_dict = None
    
#         return metrics_dict  


#     def compile(self, 
#                 optimizer :dict, 
#                 loss : Union[List[dict],Dict[str,dict]], 
#                 metrics : Optional[Union[List[dict],Dict[str,dict]]]=None,
#                 **kwargs):
#         r"""
        
#         tf_model.compile(
#                     optimizer='rmsprop',
#                     loss=None,
#                     metrics=None,
#                     loss_weights=None,
#                     weighted_metrics=None,
#                     run_eagerly=None,
#                     steps_per_execution=None,
#                     jit_compile=None,
#                     **kwargs
#                 )
#         """
#         opt = self.set_optimizer(optimizer) 
#         losses = self.build_loss_fun(loss) 
#         metrics = self.build_metric_fun(metrics)
      
#         super().compile(optimizer = opt, 
#                         loss = losses, 
#                         metrics = metrics, 
#                         **kwargs)
#         print(f"already compile model with optimizer and loss !!!!! @{self.__class__.__name__}")
