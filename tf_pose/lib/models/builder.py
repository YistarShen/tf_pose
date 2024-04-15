from lib.Registers import MODELS
from lib.Registers import LOSSES
from lib.Registers import METRICS
import copy

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
POSE_ESTIMATORS = MODELS
#LOSSES = MODELS

import tensorflow as tf

def build_backbone(cfg):
    """Build backbone."""
    #return BACKBONES.build(cfg).build_model()
    return BACKBONES.build(copy.deepcopy(cfg))

def build_neck(cfg):
    """Build neck."""
    return NECKS.build(copy.deepcopy(cfg))

def build_head(cfg):
    """Build head."""
    return HEADS.build(copy.deepcopy(cfg))

def build_pose_estimator(cfg):
    """Build pose estimator."""
    #return POSE_ESTIMATORS.build(cfg.copy())
    return POSE_ESTIMATORS.build(copy.deepcopy(cfg))

def build_loss(cfg):
    """Build LOSSES."""
    return LOSSES.build(copy.deepcopy(cfg))


def build_metric(cfg):
    """Build LOSSES."""
    return METRICS.build(copy.deepcopy(cfg))

