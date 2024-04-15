from typing import List, Optional, Sequence, Tuple, Union
import tensorflow as tf
from lib.Registers import LOSSES

############################################################################
#
#
############################################################################
@LOSSES.register_module()
class KLDiscretLoss(tf.losses.Loss):
    VERSION = '1.1.0'
    R"""KLDiscretLoss for RTMPose(RTMCCHead)
    date : 2024/1/25
    author : Dr. Shen 

    Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing.
    Modified from `the official implementation.


    Args:
        beta (float): Temperature factor of Softmax. Default: 1.0.
        label_beta (float): Temperature factor of Softmax on labels.
            Default: 10.0.
        use_label_softmax (bool): Whether to use Softmax on labels.
            Default to True.
        joints_balance_weights (List[float]): Option to use weighted loss.
            Different joint may have different target weights.
            Default to None.
        loss_weight (float) : weighted sum of loss. Default to True

    Note :
        - simcc head,  set reduction='sum'
        - tensorflow vs torch
            BATCH_SIZE = 1, K = 2, DIMS = 4
            pred = torch.rand(BATCH_SIZE,K,DIMS)
            true = torch.rand(BATCH_SIZE,K,DIMS)

            torch_pred = pred.reshape(-1, pred.size(-1))
            torch_true = true.reshape(-1, pred.size(-1))
            scores = nn.LogSoftmax(dim=1)(torch_pred)
            labels = F.softmax(torch_true, dim=1)
            kl_loss = nn.KLDivLoss(reduction='none')(scores, labels)
            print(kl_loss.mean(-1))

            tf_scores = tf.nn.softmax(pred, axis=-1)
            tf_labels = tf.nn.softmax(true, axis=-1)
            tf_kl_loss = tf.losses.KLDivergence(reduction="none")(tf_labels, tf_scores) 
            print(tf_kl_loss/DIMS)
            
    References:
        - ['KLDiscretLoss' implement @ mmpoe] (https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/classification_loss.py).
    
    
    Sample usage:

    ```python

       simcc_loss = KLDiscretLoss(
            beta =10.0, 
            label_beta : float =10.0,
            simcc_dim_splits_hw = (256*2, 192*2),
            use_label_softmax  = True,
            joints_balance_weights = [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
            ],
            loss_weight = 1.,
       )
    """

    def __init__(
        self, 
        beta : float=1.0, 
        label_beta : float =10.0,
        simcc_dim_splits_hw : Tuple[int] = (256*2, 192*2),
        use_label_softmax : bool = True,
        joints_balance_weights : Optional[List[float]]=None,
        loss_weight : float = 1.,
        **kwargs
    ) -> None:
        # simmcc head use " reduction='sum' ""
        super().__init__(**kwargs)
        self.beta = beta
        self.label_beta = label_beta
        self.use_label_softmax = use_label_softmax
        self.loss_weight = loss_weight

        if joints_balance_weights is not None and isinstance(joints_balance_weights,list):
            self.joints_balance_weights = tf.constant(
                joints_balance_weights, dtype=tf.float32
            )#(17,)
        else:
            self.joints_balance_weights = 1.

        # self.simcc_xy_dims = [
        #     int(input_size_hw[1]*simcc_split_ratio), int(input_size_hw[0]*simcc_split_ratio)
        # ]
       
        if  all([isinstance(dim_splits, int) for dim_splits in simcc_dim_splits_hw]):
            self.simcc_xy_dims = [simcc_dim_splits_hw[1], simcc_dim_splits_hw[0]]
        else:
            raise TypeError(
                "the type of simcc_dim_splits_hw must be tuple[int] or list[int] "
            )

        self.mse = tf.losses.KLDivergence(reduction="none") 

  
    def criterion(
            self, labels : tf.Tensor, dec_outs: tf.Tensor
    ) ->tf.Tensor:
        
        scores = tf.nn.softmax(dec_outs*self.beta, axis=-1)  #(b,num_joints,dims)
        if self.use_label_softmax:
            labels = tf.nn.softmax(labels*self.label_beta, axis=-1)  #(b,num_joints,dims)
        loss = self.mse(labels, scores)  #(b,num_joints,dims)=>(b,num_joints)
        return loss/tf.cast(tf.shape(labels)[-1], dtype=tf.float32)


    def call(
            self,  y_true : tf.Tensor,  y_pred : tf.Tensor
    )->tf.Tensor:
        """
        y_true : (b, 17, coord_x+coord_y)= (b, 17, simcc_xy_dims)
        y_true : (b, 17, coord_x+coord_y)
        sample_weight : (b, 17, 3)
        """
        num_of_joints = y_pred.shape[1]

        coord_pred_x, coord_pred_y = tf.split(
            y_pred, self.simcc_xy_dims, axis=-1
        ) #(b,joints,simcc_xy_dims) => (b,joints,x_dims) and (b,joints,y_dims)
        coord_true_x, coord_true_y = tf.split(
            y_true, self.simcc_xy_dims, axis=-1
        ) #(b,joints,simcc_xy_dims) => (b,joints,x_dims) and (b,joints,y_dims)

        loss = 0.
        loss = self.criterion(
            coord_true_x, coord_pred_x
        ) #(b,joints,x_dims) => (b,joints)
        loss += self.criterion(
            coord_true_y, coord_pred_y
        ) #(b,joints,y_dims) => (b,joints)
        
        loss = loss*self.joints_balance_weights*self.loss_weight #(b,joints)

        #(b,joints)
        return loss/tf.cast(num_of_joints, dtype=tf.float32) 
  


