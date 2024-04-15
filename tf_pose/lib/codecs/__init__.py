
#from .base import BaseKeypointCodec, BaseBBoxesCodec
from .base_codec import  BaseCodec
from .simcc import SimCCLabelCodec
from .yolo_AnchorBase import YoloAnchorBaseCodec
from .yolo_AnchorFree import YoloAnchorFreeCodec
from .msra_heatmaps import MSRAHeatmapCodec
from .msra_multi_heatmaps import Multi_MSRAHeatmapCodec
from .megvii_heatmaps import MegviiHeatmapCodec

#from .multi_heatmaps import MultiHeatmaps
__all__ = ['BaseCodec', 
           'YoloAnchorBaseCodec', 
           'YoloAnchorFreeCodec', 
           'SimCCLabelCodec',
           'Multi_MSRAHeatmapCodec', 
           'MSRAHeatmapCodec',
           'MegviiHeatmapCodec']