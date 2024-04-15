import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from keras import backend
from keras.utils import io_utils

class CosineDecayScheduler(tf.keras.callbacks.Callback):
    VERSION = '1.0.0'
    R""" 

    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    model.compile(tf.keras.optimizers.SGD(learning_rate=LR), loss='mse')


    TOTAL_EPOCHS = 15     
    LR = 20.0   
    lr_scheduler = CosineDecayScheduler(
        learning_rate  = LR,
        alpha = 0.5,
        total_steps = TOTAL_EPOCHS,
        warmup_steps = 0,
        hold_steps =0,
        base_decay_cycle_steps = -1,
        t_mul=1.0,
        m_mul=1.0
    )
    callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    history = model.fit(
        np.arange(100).reshape(5, 20), 
        np.zeros(5),
        epochs=TOTAL_EPOCHS, 
        callbacks=[callback], 
        verbose=2
    )     
    
    """
    def __init__(
        self,
        learning_rate : float,
        total_epochs : int,
        steps_per_epoch : int ,
        warmup_steps : int = 0,
        warmup_steps_update : int = 1, 
        hold_epochs : int =  0,
        alpha : float = 0.,
        base_decay_cycle_epochs : int = -1,
        t_mul : float =1.0,
        m_mul: float = 1.0,
        verbose = 0,
        ):
        super().__init__()


        self.verbose = verbose
        self.lr_base = float(learning_rate)
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.alpha = alpha
     
        'warmup'
        self.curr_steps = 0
        self.warmup_steps = warmup_steps
        self.warmup_steps_update = warmup_steps_update
        self.warmup_init_lr = self.alpha*self.lr_base if self.alpha else self.lr_base/100.
        self.warmup_epochs = math.ceil(warmup_steps/steps_per_epoch)
        self.by_epoch = False if self.warmup_steps>0 else True

        if total_epochs < self.warmup_epochs + hold_epochs:
            raise ValueError(
                'total_steps cannot be less than warmup_steps + hold_steps, '
                f'but got total_steps = {total_epochs} @ {self.__class__.__name__}'
            )

        'hold'
        self.hold_epochs = hold_epochs 
        'cfg of steps'
        if base_decay_cycle_epochs < 0 :
            self.base_cycle_epochs = total_epochs-self.warmup_epochs-hold_epochs
        else:
            self.base_cycle_epochs = base_decay_cycle_epochs

        if t_mul < 1.:
            raise ValueError(
                f't_mul cannot  be less than 1.0, got {t_mul} @ {self.__class__.__name__}'
            )
        if m_mul > 1.:
            raise ValueError(
                f'm_mul cannot  be greater than 1.0, got {m_mul} @ {self.__class__.__name__}'
            )     
        self.use_geometric = False  if t_mul == 1. else True
        self._t_mul = t_mul 
        self._m_mul = m_mul 
       

    def  _warmup_function(
            self,
            steps : int,  
            warmup_steps : int, 
            warmup_target_lr : float, 
            initial_lr : float, 
            dtype):
        
        total_step_delta = warmup_target_lr - initial_lr

        warmup_lr = tf.cast(
            total_step_delta * (steps / warmup_steps), dtype
        ) + initial_lr
        
        learning_rate = tf.where(
            steps >= warmup_steps, 
            warmup_target_lr, 
            warmup_lr, 
        )

        return learning_rate


    def  _decay_function(
            self, 
            steps : int, 
            decay_steps : int,  
            decay_from_lr : int,
            alpha : float,
            dtype)  :
        
        completed_fraction = tf.cast( 
           steps/ decay_steps, dtype=dtype
        ) # (105/10) = 10.5


        m_mul = tf.cast(self._m_mul, dtype)
        if self.use_geometric :

            t_mul = tf.cast(self._t_mul, dtype)
            i_restart = tf.floor(
                tf.math.log(1.0 - completed_fraction * (1.0 - t_mul))
                / tf.math.log(t_mul)
            )

            sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
            completed_fraction = (
                completed_fraction - sum_r
            ) / t_mul**i_restart

            m_fac = m_mul**i_restart
            
        else:
            i_restart = tf.floor(completed_fraction) # 10.
            completed_fraction -= i_restart # 10.5 - 10. =  0.5


        m_fac = tf.cast(m_mul**i_restart, dtype)
        #print(i_restart)
        cosine_decayed = 0.5 * m_fac*(
            1.0 
            + tf.cos(
                tf.constant(math.pi, dtype=dtype) * completed_fraction
            )
        )
        #decayed_lr = (1 - self.alpha) * cosine_decayed + self.alpha

        decayed = (1 - alpha) * cosine_decayed + alpha

        return  tf.multiply(decay_from_lr, decayed)
    
    def schedule_by_step(self, steps, lr):
        
        lr_dtype = type(lr)
        lr_base = tf.cast(
            self.lr_base , dtype=lr_dtype
        )
        warmup_init_lr = tf.cast(
            self.warmup_init_lr, dtype=lr_dtype
        )
        lr = self._warmup_function(                 
            steps,  
            warmup_steps = self.warmup_steps, 
            warmup_target_lr = lr_base, 
            initial_lr = warmup_init_lr,
            dtype = lr_dtype
        )
        return lr

    def schedule_by_epoch(self, steps, lr=None):

        lr_dtype = type(lr)
        lr_base = tf.cast(self.lr_base , dtype=lr_dtype)
        alpha = tf.cast(self.alpha , dtype=lr_dtype)

        return tf.cond(
            steps < self.warmup_epochs  + self.hold_epochs,
            lambda: lr_base,
            lambda: self._decay_function(
                    steps = (steps-self.warmup_epochs-self.hold_epochs),
                    decay_steps = self.base_cycle_epochs,
                    decay_from_lr = lr_base,
                    alpha = alpha,
                    dtype = lr_dtype,
            )
        )
    
    def on_batch_begin(self, batch, logs=None):
        if self.by_epoch :
            return 
        
        if self.curr_steps%self.warmup_steps_update!=0:
            return backend.get_value(self.model.optimizer.lr)
        
        """A backwards compatibility alias for `on_train_batch_begin`."""
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            lr = float(backend.get_value(self.model.optimizer.lr))
            lr = self.schedule_by_step(self.curr_steps, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(self.curr_steps)

        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError(
                'The output of the "schedule" function '
                f"should be float. Got: {lr}"
            )
        if isinstance(lr, tf.Tensor) and not lr.dtype.is_floating:
            raise ValueError(
                f"The dtype of `lr` Tensor should be float. Got: {lr.dtype}"
            )
        backend.set_value(self.model.optimizer.lr, backend.get_value(lr))
        
       
        
        return lr
    def on_batch_end(self, batch, logs=None):
        if self.by_epoch :
            return 
        self.curr_steps += 1

    def on_epoch_begin(self, epoch, logs=None):
        #print(self.curr_steps )
        if not self.by_epoch :
            return 
        
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            lr = float(backend.get_value(self.model.optimizer.lr))
            lr = self.schedule_by_epoch(epoch, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(epoch)

        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError(
                'The output of the "schedule" function '
                f"should be float. Got: {lr}"
            )
        if isinstance(lr, tf.Tensor) and not lr.dtype.is_floating:
            raise ValueError(
                f"The dtype of `lr` Tensor should be float. Got: {lr.dtype}"
            )
        backend.set_value(self.model.optimizer.lr, backend.get_value(lr))
        if self.verbose > 0:
            io_utils.print_msg(
                f"\nEpoch {epoch + 1}: LearningRateScheduler setting learning "
                f"rate to {lr}."
            )
        return lr
        
    def on_epoch_end(self, epoch, logs=None):
        #print(self.curr_steps )
        self.by_epoch = True if self.curr_steps >= self.warmup_steps else False
        logs = logs or {}
        logs["lr"] = backend.get_value(self.model.optimizer.lr)


    def test(self):
        decayed_lr_res = []
        self.model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
        self.model.compile(tf.keras.optimizers.SGD(self.lr_base), loss='mse')
        for epoch_ith in range(0, self.total_epochs):
            #decayed_lr = self.schedule_by_epoch(epoch_ith, lr)
            decayed_lr = self.on_epoch_begin(epoch_ith, logs=None)
            if decayed_lr is None:
                for step_ith in range(0, self.steps_per_epoch):
                    #self.schedule_by_step(self.curr_steps, lr)
                    decayed_lr = self.on_batch_begin(step_ith, logs=None)
                    self.on_batch_end(step_ith, logs=None)
            self.on_epoch_end(epoch_ith, logs=None)

            decayed_lr_res.append(decayed_lr)
            
            if epoch_ith == self.total_epochs-1:
                print(decayed_lr)
            # if step_ith == self.warmup_steps:
            #     print(f"iter={step_ith} : warmup_steps end ----------------------------------")
        del self.model

        plt.figure(figsize=(12, 6))
        plt.title("learning rate")
        plt.xlabel("epchos")
        plt.ylabel("lr")
        plt.plot(
            range(self.total_epochs), decayed_lr_res,'r', linewidth=2
        )
        plt.grid()
        plt.show()


# class CosineDecayScheduler:
#     VERSION = '1.0.0'
#     R""" 

#     model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
#     model.compile(tf.keras.optimizers.SGD(learning_rate=LR), loss='mse')
#     TOTAL_EPOCHS = 15     
#     LR = 20.0   
#     lr_scheduler = CosineDecayScheduler(
#         learning_rate  = LR,
#         alpha = 0.5,
#         total_steps = TOTAL_EPOCHS,
#         warmup_steps = 0,
#         hold_steps =0,
#         base_decay_cycle_steps = -1,
#         t_mul=1.0,
#         m_mul=1.0
#     )
#     callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

#     history = model.fit(
#         np.arange(100).reshape(5, 20), 
#         np.zeros(5),
#         epochs=TOTAL_EPOCHS, 
#         callbacks=[callback], 
#         verbose=2
#     )     
      
#     """
#     def __init__(
#         self,
#         learning_rate : float,
#         total_steps : int,
#         warmup_steps : int = 0,
#         hold_steps : int =  0,
#         alpha : float = 0.,
#         base_decay_cycle_steps : int = -1,
#         t_mul : float =1.0,
#         m_mul: float = 1.0
#         ):

#         self.lr_base = float(learning_rate)
#         self.alpha = alpha

#         if total_steps < warmup_steps + hold_steps:
#             raise ValueError(
#                 'total_steps cannot be less than warmup_steps + hold_steps, '
#                 f'but got total_steps = {total_steps} @ {self.__class__.__name__}'
#             )
        
#         'cfg of steps'
#         if base_decay_cycle_steps < 0 :
#             self.base_cycle_steps = total_steps-warmup_steps-hold_steps
#         else:
#             self.base_cycle_steps = base_decay_cycle_steps
#         self.total_steps = total_steps
#         self.warmup_steps = warmup_steps
#         self.hold_steps = hold_steps 

#         if t_mul < 1.:
#             raise ValueError(
#                 f't_mul cannot  be less than 1.0, got {t_mul} @ {self.__class__.__name__}'
#             )
#         if m_mul > 1.:
#             raise ValueError(
#                 f'm_mul cannot  be greater than 1.0, got {m_mul} @ {self.__class__.__name__}'
#             )     
#         self.use_geometric = False  if t_mul == 1. else True
#         self._t_mul = t_mul 
#         self._m_mul = m_mul 

        
        
#     def  _warmup_function(
#             self,
#             steps : int,  
#             warmup_steps : int, 
#             warmup_target_lr : float, 
#             initial_lr : float, 
#             dtype):
        
#         total_step_delta = warmup_target_lr - initial_lr

#         warmup_lr = tf.cast(
#             total_step_delta * (steps / warmup_steps), dtype
#         ) + initial_lr
        
#         learning_rate = tf.where(
#             steps > warmup_steps, 
#             warmup_target_lr, 
#             warmup_lr, 
#         )

#         return learning_rate


#     def  _decay_function(
#             self, 
#             steps : int, 
#             decay_steps : int,  
#             decay_from_lr : int,
#             alpha : float,
#             dtype)  :
        
#         completed_fraction = tf.cast( 
#            steps/ decay_steps, dtype=dtype
#         ) # (105/10) = 10.5


#         m_mul = tf.cast(self._m_mul, dtype)
#         if self.use_geometric :

#             t_mul = tf.cast(self._t_mul, dtype)
#             i_restart = tf.floor(
#                 tf.math.log(1.0 - completed_fraction * (1.0 - t_mul))
#                 / tf.math.log(t_mul)
#             )

#             sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
#             completed_fraction = (
#                 completed_fraction - sum_r
#             ) / t_mul**i_restart

#             m_fac = m_mul**i_restart
            
#         else:
#             i_restart = tf.floor(completed_fraction) # 10.
#             completed_fraction -= i_restart # 10.5 - 10. =  0.5

#         if steps==200:
#            print(completed_fraction)    

#         m_fac = tf.cast(m_mul**i_restart, dtype)
#         #print(i_restart)
#         cosine_decayed = 0.5 * m_fac*(
#             1.0 
#             + tf.cos(
#                 tf.constant(math.pi, dtype=dtype) * completed_fraction
#             )
#         )
#         #decayed_lr = (1 - self.alpha) * cosine_decayed + self.alpha

#         decayed = (1 - alpha) * cosine_decayed + alpha

#         return  tf.multiply(decay_from_lr, decayed)
    
    
#     def __call__(self, steps,lr):
#         lr_dtype = type(lr)
#         lr_base = tf.cast(self.lr_base , dtype=lr_dtype)
#         alpha = tf.cast(self.alpha , dtype=lr_dtype)

#         return tf.cond(
#             steps < self.warmup_steps + self.hold_steps,
#             lambda: self._warmup_function(
#                     steps,  
#                     warmup_steps = self.warmup_steps, 
#                     warmup_target_lr = lr_base, 
#                     initial_lr = self.lr_base/10.,
#                     dtype = lr_dtype,
#             ),
#             lambda: self._decay_function(
#                     steps = (steps-self.warmup_steps-self.hold_steps),
#                     decay_steps = self.base_cycle_steps,
#                     decay_from_lr = lr_base,
#                     alpha = alpha,
#                     dtype = lr_dtype,
#             )
#         )
    
#     def test(self):
#         decayed_lr_res = []
#         for step_ith in range(0, self.total_steps):
#             decayed_lr = self(step_ith, self.lr_base)
#             decayed_lr_res.append(decayed_lr)
#             #print(f"decayed_lr {step_ith}th: ", decayed_lr)

#             if step_ith== self.total_steps:
#                 print(decayed_lr)
#             if step_ith == self.warmup_steps:
#                 print(f"iter={step_ith} : warmup_steps end ----------------------------------")
                 
#         plt.figure(figsize=(12, 6))
#         plt.title("learning rate")
#         plt.xlabel("epchos")
#         plt.ylabel("lr")
#         plt.plot(
#             range(self.total_steps),decayed_lr_res,'r', linewidth=2
#         )
#         plt.show()