# from lib.datasets.builder import  (pose_dataloader, det_dataloader)
from lib.datasets.tfds_builder import  dataloader
from .tfrec_parsers import *
__all__ = ['dataloader']

