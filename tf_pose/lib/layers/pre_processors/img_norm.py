
import tensorflow as tf
from lib.Registers import LAYERS


@LAYERS.register_module()
class ImgNormalization(tf.keras.layers.Layer):
    VERSION ='1.0.0'
    R""" 
    
    """
    def __init__(self,
            img_mean = [0.485, 0.456, 0.406],  
            img_std = [0.229, 0.224, 0.225],         
            **kwargs):
        super().__init__(**kwargs)

        self.img_mean = img_mean
        self.img_std = img_std
        self.img_var = [std**2 for std in img_std]

    def build(self, input_shape):
        self.n, self.h, self.w, self.c = input_shape.as_list()
        assert self.c==3,'input must be (b, h, w, 3)@ImgPreProcess_Layer'
        #self.img_var= tf.math.square(self.img_std)  
        self.img_norm_layer = tf.keras.layers.Normalization(
            axis=-1, 
            mean=self.img_mean,  
            variance=self.img_var
        )
    def call(self, x):
        '''
        x : (b,h,w,3)
        '''
        img = tf.cast(x, dtype=tf.float32)/255.
        img_norm = self.img_norm_layer(img)
        return img_norm
        
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "img_mean": self.img_mean,
                "img_std" : self.img_std, 
                "img_var" : self.img_var,       
            }
        )
        return config 

