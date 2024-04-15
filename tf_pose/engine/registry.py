import importlib
import inspect
import logging
#https://colab.research.google.com/github/open-mmlab/mmpose/blob/master/demo/MMPose_Tutorial.ipynb#scrollTo=WR9ZVXuPFy4v
#https://applenob.github.io/python/register/
from collections.abc import Callable
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union
from lib.utils.common import is_path_available


class Registry:
    def __init__(self, 
                registry_name : str,
                build_func: Optional[Callable] = None,
                locations: List = [],
                show_log : Optional[bool] = False):
        
        from .build_functions import build_from_cfg
        self._module_dict : Dict[str, Type] = dict()

        self._imported = False
       
        self._name = registry_name

        self._locations = locations

        self.build_func: Callable
        if build_func is not None :
           self.build_func =  build_func
        else:
            #defualt build_from_cfg
            self.build_func =  build_from_cfg

        self._show_log = show_log

    def __setitem__(self, module_name, module):
        if not callable(module):
            raise Exception(f"Value of a Registry must be a callable")
        if module_name is None:
            module_name = module.__name__
        #modules had the same names
        if module_name in self._module_dict:
            logging.warning("Key '%s' already in registry '%s'." % (module_name, self._name))
        self._module_dict[module_name] = module
        print(f'key : {module_name}, value :{module} @update dict')


    def register(self,    
                module :Optional[Type] = None) -> Union[type, Callable]:
        'target'
        ''

        #print(module.__name__,"test",  callable(module) )
        if not inspect.isclass(module) and not inspect.isfunction(module):
            raise TypeError('module must be a class or a function, '
                            f'but got {type(module)}')
        
    
        """Decorator to register a function or class."""
        def decorator(key, value):
            self[key] = value
            return value
        
        if callable(module):
            return decorator(None, module)
        
        return lambda x: decorator(module, x)
    
    def _register_module(self,
                         module: Type,
                         module_name: Optional[Union[str, List[str]]] = None,
                         force: bool = False) -> None:
        
        if not inspect.isclass(module) and not inspect.isfunction(module):
            raise TypeError('module must be a class or a function, '
                            f'but got {type(module)}')
        
        if not callable(module):
            raise Exception(f"Value of a Registry must be a callable")
        
        if module_name is None:
            module_name = module.__name__
        '''
        if not force and module_name in self._module_dict:
            existed_module = self._module_dict[module_name]
            raise KeyError(f'{module_name} is already registered in {self._name} '
                               f'at {existed_module.__module__}')
        if force and module_name in self._module_dict:
            logging.warning("Key %s already in registry %s." % (module_name, self.__name__))

        '''
        if module_name in self._module_dict:
            if not force :
                existed_module = self._module_dict[module_name]
                raise KeyError(f'{module_name} is already registered in {self._name} '
                                f'at {existed_module.__module__}')
            else:
                logging.warning("Key %s already in registry %s." % (module_name, self.__name__))

        self._module_dict[module_name] = module
        if self._show_log :
            print(f'key : {module_name}, value :{module} >')

    def register_module(self,
                        name: Optional[Union[str, List[str]]] = None,
                        force: bool = False,
                        module: Optional[Type] = None) -> Union[type, Callable]:
        
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')
        
        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(
                'name must be None, an instance of str, or a sequence of str, '
                f'but got {type(name)}')
              
        
        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module
        
        # use it as a decorator: @x.register_module()       
        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register
    

    def import_from_location(self) ->None:
        """import modules from the pre-defined locations in self._location."""
        if not self._imported :
            print(f'register : <{self._name}>  init ...................from internal libs\n') 
            for loc in self._locations:
                #is_path_avaiable(loc)
                try:
                   importlib.import_module(loc)
                   print(
                        f"\n All Modules of registry < {self._name} >have "
                        f'been automatically imported from {loc}\n')                
                except (ImportError, AttributeError, ModuleNotFoundError):
                    print(
                        f'Failed to import {loc}, please check the '
                        f'location of the registry < {self._name} > is correct.\n')
                    
            print(f'--------------register : <{self._name}>  Update Done-----from internal libs\n')      
        self._imported = True

 
        
    def get(self, key: str) ->Optional[Type]:
        
        self.import_from_location()
        '''
        if self._show_log :
           print(self._module_dict) 
        '''
        obj_cls = self._module_dict[key]
        return obj_cls
    
    def build(self, cfg: dict, *args, **kwargs) -> Any:
        return self.build_func(cfg, *args, **kwargs, registry=self)
    
    @property
    def name(self):
        return self._name
    
    @property
    def module_dict(self):
        return self._module_dict
       
    def __len__(self):
        return len(self._module_dict)
    
    def __getitem__(self, key):
        return self._module_dict[key]

    def __contains__(self, key):
        return key in self._module_dict
    '''
    to do
    def __repr__(self):
    '''

    def keys(self):
        return self._module_dict.keys()
    








class Registers:
    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")
    
    model_test = Registry('model_test', locations=['lib.pkg_sample'] ) #

