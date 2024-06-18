# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_cat_seg_config

# dataset loading
from .data.dataset_mappers.detr_panoptic_dataset_mapper import DETRPanopticDatasetMapper
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# models
from .cat_seg_model import CATSeg
from .test_time_augmentation import SemanticSegmentorWithTTA
from .implicit_fusion_Ver02 import ImplicitFusionCATSegVer02
from .implicit_fusion_Ver03 import ImplicitFusionCATSegVer03
from .implicit_fusion_Ver04 import ImplicitFusionCATSegVer04
from .implicit_fusion_Ver05 import ImplicitFusionCATSegVer05
from .vision_transformer import vit_base
# from .vision_transformer import * 
# from .vision_transformer import vits