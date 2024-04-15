from .simcc_loss import KLDiscretLoss
from .yolo_anchor_base_loss import YoloAnchorBaseLoss
from .yolo_anchor_free_loss import YoloAnchorFreeBBoxLoss, YoloAnchorFreeClassLoss
from .heatmap_loss import KeypointsMSELoss, MultiHeatmapMSELoss
from .lifter_loss import PoseLifterRegLoss
__all__ = [
        'KLDiscretLoss',
        'YoloAnchorBaseLoss', 'YoloAnchorFreeBBoxLoss', 'YoloAnchorFreeClassLoss',
        'PoseLifterRegLoss' , 'KeypointsMSELoss', 'MultiHeatmapMSELoss'
]

#YoloAnchorBaseLoss