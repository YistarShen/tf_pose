
import inspect
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union
from .registry import Registry

''' 
dict, BaseModule, BaseBackbone, tf.keras.Layers
'''
def build_from_cfg(cfg :dict, 
                   registry : Registry,
                   show_log : Optional[bool]=False) ->Any:

    if not isinstance(cfg, dict):
        raise TypeError(
            f'cfg must be a dict, but got {type(cfg)}')
    
    if 'type' not in cfg:
        raise KeyError(
            ' `cfg`  must contain the key "type", 'f'but got {cfg}\n')   
    
    if not isinstance(registry, Registry):
        raise TypeError('registry must be a engine.Registry object, '
                        f'but got {type(registry)}')
    
    if not isinstance(show_log, bool):
        raise TypeError(f'show_log must be a boolean, but got {type(show_log)}')
    
    'copy cfg'
    args = cfg.copy()
    'get type'
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None :
            raise KeyError(
                    f'{obj_type} is not in the {registry.name} registry. '
                    f'Please check whether the value of `{obj_type}` is '
                    'correct or it was registered as expected. '
                )  
    elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
        'support obj_type is class or function not only str'
        obj_cls = obj_type     
    else:
        raise TypeError(
            'type must be a str, class or function type, '
            f'but got {type(obj_type)}'
        )
    
    'init'
    try:
        '''
        if inspect.isclass(obj_cls):  # type: ignore
            obj = obj_cls.get_instance(**args)  # type: ignore
        else:
            obj = obj_cls(**args) 
            
        '''
        if inspect.isfunction(obj_cls):
            print("fun return Callable function")
            return obj_cls(**args)

        obj = obj_cls(**args) 

        if show_log :
            print(f'the `{obj_cls.__name__}` instance is built from ' 
                f'register <{Registry._name}>, its implementation can be found in  {obj_cls.__module__}')
        
        return obj
    
    except Exception as e:
        print(f"fail @build_from_cfg : <{cfg}>")

