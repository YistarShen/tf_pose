import os
from collections.abc import Callable 
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from tensorflow import Tensor
import tensorflow as tf
from tensorflow.keras.layers import  Input
from tensorflow.keras.models import Model
from .base import BaseModel
from lib.Registers import MODELS
from lib.models.builder import build_backbone, build_neck,build_head, build_loss, build_metric

@MODELS.register_module()
class LifterPoseEstimator(BaseModel):
    def __init__(self,
        backbone: dict = None,
        neck: dict = None,
        head: dict = None,
        extract_layres : Optional[List[str]] = [],
        optimizer : dict = None,
        metrics : Optional[Union[List[dict], dict]] = None,
        weighted_metrics : Optional[Union[List[dict], dict]] = None,
        losses :  Union[List[dict],Dict[str,dict]] = None,
        pretrained_weights_path :  Optional[str] = None,
        name : Optional[str] = None):
        
        super().__init__(
            backbone = backbone,
            neck = neck,
            head = head,
            extract_layres = extract_layres,
            name = name)
        
    
        if pretrained_weights_path is not None:
           if os.path.exists(pretrained_weights_path) :
                self.tf_model.load_weights(pretrained_weights_path, by_name=True, skip_mismatch=False)
                print(f"already load weights from {pretrained_weights_path} with [ by_name=True, skip_mismatch=False ]")
           else:    
                print(f"Warning!!!!!!! Cannot find pretrained weights in {pretrained_weights_path}")


        'set enable metrics/weighted_metrics '
        self.use_weighted_metrics =  False
        if weighted_metrics is not None:
           self.use_weighted_metrics =  True
           metrics = weighted_metrics
   

        'compile model with optimizer, loss_fn and metric_fn, here optimizer and loss_fn is necessary(optinal) '
        if losses is not None and optimizer is not None :
            self.model_compile(optimizer_cfg = optimizer,
                                losses_cfg = losses,
                                metrics_cfg = metrics)
    
    def set_optimizer(self,
                    optimizer_cfg : Optional[dict]=None) : 
        opt_cfg = optimizer_cfg.copy()
        if isinstance(optimizer_cfg,dict): 
            opt = getattr(tf.keras.optimizers, opt_cfg.pop('type') )(**opt_cfg)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        return opt
    

    def build_loss_fun(self,
                    losses_cfg : Union[List[dict],Dict[str,dict]]) -> Dict[str, tf.losses.Loss] : 
        
        assert len(losses_cfg) == len(self.tf_model.outputs), \
                f'loss_fn num is {len(losses_cfg)}, but num of model_outputs is {len(self.tf_model.outputs)} @build_loss_fun'
        
        losses_dict = dict()
        if isinstance(losses_cfg, dict):
            'dict'
            losses_dict = losses_cfg.copy()
            for (k,v), output_layer_name in zip(losses_cfg.items(),self.tf_model.output_names):
                assert k == output_layer_name, \
                f"losses_cfg key : {k}  doesn't match  model_output_name : {output_layer_name}"
                losses_dict[k] =  build_loss(v) 
        
        elif isinstance(losses_cfg, list):
            'list'
            for k,v in zip(self.tf_model.output_names, losses_cfg) :
                losses_dict[k] = build_loss(v) if isinstance(v, dict) else tf.keras.losses.MeanSquaredError(name='mse')

        else:
            'other type'
            for k in self.tf_model.output_names:
                losses_dict[k] = tf.keras.losses.MeanSquaredError(name='mean_squared_error')
            #raise RuntimeError("type of losses_cfg is invalid , losses_cfg must be ( List | Dict )")   
        return losses_dict 
    

    def build_metric_fun(self,
                        metrics_cfg : Optional[Union[List[dict],Dict[str,dict]]]=None) -> Union[Dict[str, tf.keras.metrics.Metric], None] : 
        
        metrics_dict = dict()
        if isinstance(metrics_cfg, list):
            'list'
            assert len(metrics_cfg) == len(self.tf_model.outputs), \
                  f'len(metrics_cfg) : {len(metrics_cfg)} must be len(tf_model.outputs) : {len(self.tf_model.outputs)} if metrics_cfg is a List @build_loss_fun'
            
            'list'
            for k,v in zip(self.tf_model.output_names, metrics_cfg) :
                metrics_dict[k] = build_metric(v)

        elif isinstance(metrics_cfg, dict):
            'dict'
            metrics_dict = metrics_cfg.copy()
            for k,v in metrics_cfg.items():
                if k not in self.tf_model.output_names:
                    raise RuntimeError(f"metrics_cfg.key : {k}  must be in {self.tf_model.output_names} if metrics_cfg is a dict ")   
                else:
                    metrics_dict[k] = build_metric(v) 
        else:
            'other type'
            metrics_dict = None

        return metrics_dict
   
    def model_compile(self, 
                    optimizer_cfg :dict, 
                    losses_cfg : Union[List[dict],Dict[str,dict]], 
                    metrics_cfg : Optional[Union[List[dict],Dict[str,dict]]]=None):
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
        opt = self.set_optimizer(optimizer_cfg) 
        losses = self.build_loss_fun(losses_cfg) 
        metrics = self.build_metric_fun(metrics_cfg)
     
        self.tf_model.compile(optimizer = opt, 
                              loss = losses, 
                              metrics = metrics if not self.use_weighted_metrics else None,
                              weighted_metrics = metrics if self.use_weighted_metrics else None)


    def get_model(self):
        return self.tf_model
    
    def get_backbone(self):
        return super().get_backbone()
    
    def get_head(self):
        return super().get_head()
        
    def preprocss_model(self):
        return NotImplemented
    
    def postprocess_model(self):
        return NotImplemented
    
    def load_weights(self):
        return NotImplemented
    
    def compile(self):
        return NotImplemented