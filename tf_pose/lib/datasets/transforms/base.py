
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Sequence
from tensorflow import Tensor
import tensorflow as tf


class CVBaseTransformLayer(tf.keras.layers.Layer, metaclass=ABCMeta):
    VESRION = '3.0.0'
    R"""CVBaseTransform Layer.
    date : 2024/3/24
    author : Dr. Shen 

    1. data type must be dict[str,Tensor] ,dict[str,RaggedTensor], 
        dict[str,dict[str,RaggedTensor]] or dict[str,dict[str,Tensor]] 
    2. support batch tf.RaggedTensor and batch tf.Tensor
    3. to keep input and output format is same, 
        if type(input)=tf.RaggedTensor, type(output) will be tf.RaggedTensor
    
    
    """
    def __init__(
            self, 
            parallel_iterations : int = 16,
            test_mode : bool = False,
            **kwargs
        ):
        super().__init__(trainable =False, **kwargs)
        
        self.parallel_iterations = parallel_iterations
        self.test_mode = test_mode
        self.fp_zero = tf.cast(0., dtype=self.compute_dtype)
        self.fp_one = tf.cast(1., dtype=self.compute_dtype)
        # if isinstance(required_keys,list) and all([ isinstance(key, str) for key in required_keys]):
        #     print()
        

    
    def verify_required_keys(
            self, data : Dict, keys: List[str]=['image']
    ):
        """
        in general, 'image' is required key for most cv transforms
        because we need to use data['image'].ndims to verify whether data is batched
        """
        for key in keys:
            assert data.get(key,None) is not None, \
            f"Require Data Key : {key}  @{self.__class__.__name__}"
   
    def with_meta_info(self, data : Dict):
        return False if data.get('meta_info', None) is None else True
        
    def with_transform2src(self, data : Dict):
        return False if data.get('transform2src', None) is None else True
       
    def img_to_tensor(self, 
                    image : Union[tf.Tensor, tf.RaggedTensor],
                    dtype = tf.uint8) -> tf.Tensor:
        """
        most image ops require image format must be tf.Tensor
        so if image is ragged tensor, we convert it to tf.Tensor 
        """
        if not isinstance(image, tf.Tensor):
           image =  image.to_tensor()
        if image.dtype!=dtype:
           image = tf.cast(image, dtype=dtype)
        return image
    
    
    def update_data_img(self, 
                        img_tensor : tf.Tensor,
                        data : Dict[str,Union[tf.Tensor, tf.RaggedTensor]]) -> Dict[str,Union[tf.Tensor, tf.RaggedTensor]]:
        """
        after using image ops to do transform, image is tf.Tensor.
        but we want to keep image output format same as image input format
        so if input image is ragged tensor, we convert it back to tf.ragged_tensor from tf.Tensor
        """
        if data['image'].dtype != img_tensor.dtype :
            img_tensor = tf.cast(img_tensor, dtype=data['image'].dtype)

        if isinstance(data['image'], tf.RaggedTensor):
            data['image'] = tf.RaggedTensor.from_tensor(img_tensor)
        else:
            data['image'] = img_tensor
    
        if data.get('image_size', None) is not None:
            data['image_size'] = tf.cast(
                    tf.shape(img_tensor)[:2], dtype=tf.int32
            )
        return data


    
    # def compute_signature(self,
    #                       data, 
    #                       fn_output_signature = {}):
    #     """
    #     recursive method to find all output signatures for tf.map_fn
    #     """
    #     for key, val in data.items():
    #         if isinstance(val, dict):
    #             sub_fn_output_signature = self.compute_signature(val,{})
    #             fn_output_signature[key] = sub_fn_output_signature
    #         else:
    #             #print(key, "-----", tf.type_spec_from_value(val[...]))
    #             if isinstance(val, tf.RaggedTensor):
    #                 TensorSpec = tf.type_spec_from_value(val)
    #                 if TensorSpec.ragged_rank>=2 :
    #                     fn_output_signature[key] = tf.RaggedTensorSpec(
    #                         shape=TensorSpec.shape[1:],
    #                         ragged_rank=TensorSpec.ragged_rank-1,
    #                         dtype=TensorSpec.dtype
    #                     )  
    #                     'image'
    #                     #fn_output_signature[key] = tf.type_spec_from_value(val[0,...])
    #                 else:
    #                     'bboxes, labels, kps'
    #                     fn_output_signature[key] = tf.RaggedTensorSpec(
    #                                             shape=val.shape[1:],
    #                                             ragged_rank=0,
    #                                             dtype=TensorSpec.dtype
    #                     )  
    #             elif isinstance(val, tf.Tensor):   
    #                 TensorSpec = tf.type_spec_from_value(val[0,...])
    #                 fn_output_signature[key] = TensorSpec   
    #             else:
    #                 raise TypeError(
    #                     "type of value in data(dict) must be tf.Tensor or tf.RaggedTensor"
    #                      f", but got {type(val)} for key = '{key}")    
        
    #     return  fn_output_signature
    
    
    
    def _compute_signature(
            self, data
    ):
        """
        recursive method to find all output signatures for tf.map_fn
        """
        
        # fn_output_signature = tf.nest.map_structure(
        #     lambda x : tf.type_spec_from_value(x[0]) , data
        # )
        TensorSpec = tf.nest.map_structure(
            lambda x : tf.type_spec_from_value(x) , data
        )
        if not all([isinstance(spec,(tf.RaggedTensorSpec,tf.TensorSpec)) for spec in tf.nest.flatten(TensorSpec)]):
            raise TypeError(
                "the type of values in data(dict) must be tf.Tensor or tf.RaggedTensor"
                f", but got {TensorSpec}, please check content of data"
            )    
        
        fn_output_signature = tf.nest.map_structure(
            lambda x : tf.RaggedTensorSpec(
                    shape=x.shape[1:],
                    ragged_rank=x.ragged_rank-1,
                    dtype=x.dtype
            )
            if isinstance(x, tf.RaggedTensorSpec)
            else
            tf.TensorSpec(
                shape = x.shape[1:],
                dtype=x.dtype,
            ) ,
            TensorSpec
        )
        return  fn_output_signature
    

        

    def count_params(self):
        # The label encoder has no weights, so we short-circuit the weight
        # counting to avoid having to `build` this layer unnecessarily.
        return 0  
    
    
    def _rand_truncated_normal(self,shape=(), mean=0.0, stddev=0.5):
        return tf.random.truncated_normal(
            shape=shape,
            mean=mean,
            stddev=stddev,
            dtype=self.compute_dtype
        )
    
    def _rand_val(self, shape=(),  minval=-1., maxval=1., seed=None):
        return tf.random.uniform(
            shape=shape,  
            minval=minval,
            maxval=maxval, 
            seed=seed, 
            dtype= self.compute_dtype 
        )

    def _rand_prob(self, shape=(),  seed=None):
        '可以傷除, 使用 _rand_val' 
        return self._rand_val(
            shape=shape,  minval=0.,maxval=1.,  seed=seed
        )

    def _rand_inverse(
            self, prob : float =0.5, shape=(), seed=None
    ):
        val =  self._rand_val(shape=shape, minval=0., maxval=1.,  seed=seed)
        if shape==() :
            #scalar
            return tf.cond(
                prob > val, lambda : self.fp_one , lambda:  -self.fp_one
            )
        #tensor
        return tf.where(
            tf.greater(prob, val), self.fp_one , -self.fp_one
        ) 


    def _rand_bool(
            self, prob : float =0.5, shape=()
    ):
        #scalar
        if shape==():
            return tf.cond(
                tf.cast( prob, dtype=self.compute_dtype)>self._rand_val(shape=(), minval=0., maxval=1.), 
                lambda : True, 
                lambda : False
            )
        #tensor
        return tf.where(
            tf.greater(prob, self._rand_val(shape=shape, minval=0., maxval=1.)),
            True, 
            False
        )

    def _op_ensure_fp_dtype(self, fp_tensor):
        '可以傷除' 
        return tf.cast(fp_tensor, dtype=self.compute_dtype)
    

    def _op_expand_batch_dim(
            self, data : Union[Tensor, Sequence[Tensor], Dict[str,Tensor]]
    ):
        return tf.nest.map_structure(
            lambda x : tf.expand_dims(x, axis=0), data
        )
    
    def _op_squeeze_batch_dim(
            self, data : Union[Tensor, Sequence[Tensor], Dict[str,Tensor]]
    ):
        return tf.nest.map_structure(
            lambda x : tf.squeeze (x, axis=0), data
        )  
    def _op_copy_dict_data(self, data : dict):
        if not isinstance(data, dict):
            raise TypeError(
                "op_copy_dict_data only support 'dict' type, "
                f", but got {type(data)}"
            )    
        return {k:v for k,v in data.items()}

    def _op_cast(
        self, 
        data : Union[Tensor, Sequence[Tensor], Dict[str,Tensor]],
        dtype = tf.float32
    ):
        return  tf.nest.map_structure(
            lambda x : tf.cast(x, dtype=dtype), data
        )
    
    def _op_to_tensor(
            self, data : Union[Tensor, Sequence[Tensor], Dict[str,Tensor]]
    ):
        return tf.nest.map_structure(
            lambda x : x.to_tensor(), data
        )  
    
    def transform(
            self,data: Dict
    ) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function. All subclass of BaseTransform should
        override this method.
        This function takes the result dict as the input, and can add new
        items to the dict or modify existing items in the dict. And the result
        dict will be returned in the end, which allows to concate multiple
        transforms into a pipeline.
        Args:
            results (dict): The result dict.
        Returns:
            dict: The result dict.
        """
        raise NotImplementedError()
    
    def batched_transform(self,data: Dict,  *args, **kwargs):
        
        data = tf.map_fn(
            lambda x : self.transform(x[0],*x[1:], **kwargs),
            (data,*args), 
            parallel_iterations = self.parallel_iterations,
            fn_output_signature = self._compute_signature(data)
        )
        return data

    @tf.function
    def call(self, data : Dict, *args, **kwargs) ->Dict:
        """ it can fully support batched transform 
        1. only support dict[str, (tf.Tensor or tf.RaggedTensor) ] format
        2. if additional data keys will be generated by transform, 
           need to init new keys for data dict before dooing transform
        3. use image keys to verfiy whether data is batched
           non batch data : data['image'].shape.rank = 3
           batch data : data['image'].shape.rank = 4

            data = tf.map_fn(
                lambda x : self.transform(x, **kwargs),
                data, 
                parallel_iterations = self.parallel_iterations,
                fn_output_signature = self.compute_signature(data,{})
            )

           def test(data,*args,**kwargs):
                data = tf.map_fn(
                    lambda x : transform(x[0], *x[1:], **kwargs),
                    (data, *args),
                    fn_output_signature = tf.TensorSpec((8), dtype=tf.int32),
                )
                return data

        """
        if not isinstance(data,dict):
            raise TypeError(
                "data type must be dict[str,Tensor],"
                f" but got {type(data)}@{self.__class__.__name__}"
            )
        
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!",data['image'].shape)
        if hasattr(self,'add_Datakeys'):
            data = self.add_Datakeys(data)

        
        if data['image'].shape.rank == 3:
            data = self._op_expand_batch_dim(data)
            data = self.batched_transform(data, *args, **kwargs)
            data = self._op_squeeze_batch_dim(data)
        elif  data['image'].shape.rank == 4:
            data = self.batched_transform(data, *args, **kwargs)        
        else:
            raise ValueError(
                "data['image'].shape.rank must be 3 or 4"
                f"but got {data['image'].shape.rank} @{self.__class__.__name__}" 
            )
        return data
    

    


class VectorizedTransformLayer(CVBaseTransformLayer):
    VESRION = '1.0.0'
    R"""CVBaseTransform Layer.
    date : 2024/3/24
    author : Dr. Shen 


    """
    def __init__(
            self, **kwargs
        ):
        super().__init__(**kwargs)
        "Vectorized type doesn't beed these attrs"
        # self.__delattr__('img_to_tensor') # # convert single ragged image to tensor
        # self.__delattr__('update_data_img') # update single image 
        # self.__delattr__('transform')  # single sample transform

    def batched_transform(self,data: Dict,  *args, **kwargs):
        raise NotImplementedError()
    
    @tf.function
    def call(self,dic : Dict,  *args, **kwargs) ->Dict:
        #data = {k:v for k,v in dic.items()}
        data = self._op_copy_dict_data(dic) #must copy data ???
        del dic

        if not isinstance(data,dict):
            raise TypeError(
                "data type must be dict[str,Tensor],"
                f" but got {type(data)}@{self.__class__.__name__}"
            )
        
        if data['image'].shape.rank == 4:
            data = self.batched_transform(data)
        else:
            raise ValueError(
                "cls : VectorizedTransformLayer must receive batched data "
                f"but got {data['image'].shape.rank} @{self.__class__.__name__}"
                "Please call this transform by sending samples with data['image'].shape.rank=4 "
            )
        return data 
        