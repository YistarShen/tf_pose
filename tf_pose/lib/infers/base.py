
from collections.abc import Callable 
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from tensorflow import Tensor
import tensorflow as tf
from tensorflow.keras.layers import  Input
from tensorflow.keras.models import Model
import os 
##############################################################################################
#
#
##############################################################################################

class BaseProc(metaclass=ABCMeta):
    def __init__(self,  
                batch_size  : int=1,
                name : Optional[str] = None ):


        self.model_name = name
        self.tf_model_save_path = None
        self.batch_size = batch_size


        '#1 Set Inputs Tensor Spec'
        self.InputsTensor = self.Set_InputsTensorSpec(self.batch_size)
        '#2 build infer proc as tf-model'
        self.model = self.build_model(self.InputsTensor)

        self.bool = True
    
    @abstractmethod
    def Set_InputsTensorSpec(self, 
                             batch_size :int=1): 
        """Set_InputsTensorSpec.
        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """
        return NotImplemented
             
 
    @abstractmethod
    def forward(self, 
                x :  Union[List[Tensor],Tuple[Tensor]]) -> Union[List[Tensor],Tuple[Tensor]]:
        """Forward function.
        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """
        return NotImplemented

    @abstractmethod    
    def __call__(self, data: Tensor) -> Union[Tensor,Tuple[Tensor]] :
        return self.forward(data)    
 
    def build_model(self, 
                    x : List[Tensor], 
                    save_model : bool = False) ->object:

        """Forward function.
        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """
        return Model(inputs = x, outputs = self.forward(x))
    
    def get_tf_model(self):
        return self.model
    
    def get_InputsTensorSpec(self):
        return self.model.inputs  
    
    def get_OutputsTensorSpec(self):
        return self.model.outputs  
    
    def save_model(self, 
                   saved_model_dir, 
                   model_name=None,
                   load_saved_model=False):
        
        assert os.path.exists(saved_model_dir)==True, \
        f"cannot find MODEL_LOAD_DIR @BaseProc.save_model: \n<{saved_model_dir}>\n" 

        if model_name is None :
            self.tf_model_save_path = os.path.join(saved_model_dir, self.model_name)
        else:
            if isinstance(model_name,str):
                self.tf_model_save_path = os.path.join(saved_model_dir, model_name)
            else:
                raise RuntimeError(f"ivalid type !!!! type of model_name must be str, @BaseProc.save_model")  

        'SAVE MODEL'
        #non_compiled_model.save(tf_model_save_path)
        tf.keras.models.save_model(self.build_model(),
                                self.tf_model_save_path,
                                overwrite=True,
                                include_optimizer=False,
                                save_format='tf',
                                signatures=None,
                                options=None,
                                save_traces=True
        )

        if os.path.exists(self.tf_model_save_path):
            print(f"already generate tf model : \n<{self.tf_model_save_path}>\n")    
        else:
            raise ValueError("fail to generate TF MODEL  !!!!!@BaseProc.save_model") 
        
        if load_saved_model :
            self.model = tf.keras.models.load_model(self.tf_model_save_path, compile=False)
        
    def load_model(self, model_saved_path):

        if isinstance(model_saved_path,str):
            self.tf_model_save_path = model_saved_path
        else:
            raise RuntimeError(f"ivalid type !!!! type of model_save_path must be str, @BaseProc.load_model")  
        
        assert os.path.exists(self.tf_model_save_path)==True, f"cannot find model_save_path @BaseProc.load_model: \n<{self.tf_model_save_path}>\n" 

        self.model = tf.keras.models.load_model(self.tf_model_save_path, compile=False)

        return self.model

    def get_saved_model_path(self):
        return self.tf_model_save_path 


