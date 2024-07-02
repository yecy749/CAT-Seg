# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union
from einops import rearrange

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer.cat_seg_predictor import CATSegPredictor
from ..transformer.FusionPredictorVer07 import FusionPredictorVer07
from ..transformer.FusionPredictorVer08 import FusionPredictorVer08
from ..transformer.FusionPredictorVer09 import FusionPredictorVer09
from ..transformer.FusionPredictorVer09a import FusionPredictorVer09a
from ..transformer.FusionPredictorVer09b import FusionPredictorVer09b
from ..transformer.FusionPredictorVer09c import FusionPredictorVer09c
from ..transformer.FusionPredictorVer09d import FusionPredictorVer09d
from ..transformer.FusionPredictorVer09e import FusionPredictorVer09e
from ..transformer.FusionPredictorVer10 import FusionPredictorVer10
from ..transformer.FusionPredictorVer11 import FusionPredictorVer11
from ..transformer.FusionPredictorVer12 import FusionPredictorVer12
from ..transformer.FusionPredictorVer12a import FusionPredictorVer12a
from ..transformer.FusionPredictorVer13 import FusionPredictorVer13
@SEM_SEG_HEADS_REGISTRY.register()
class FusionHeadVer12a(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": FusionPredictorVer12a(
                cfg,
            ),
        }

    def forward(self, features,dino_feat, guidance_features, dino_guidance_feat, prompt=None, gt_cls=None):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat,dino_feat, guidance_features,dino_guidance_feat, prompt, gt_cls)

@SEM_SEG_HEADS_REGISTRY.register()
class FusionHeadVer13(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": FusionPredictorVer13(
                cfg,
            ),
        }

    def forward(self, features,dino_feat, guidance_features, dino_guidance_feat, prompt=None, gt_cls=None):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat,dino_feat, guidance_features,dino_guidance_feat, prompt, gt_cls)


@SEM_SEG_HEADS_REGISTRY.register()
class FusionHeadVer12(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": FusionPredictorVer12(
                cfg,
            ),
        }

    def forward(self, features,dino_feat, guidance_features, dino_guidance_feat, prompt=None, gt_cls=None):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat,dino_feat, guidance_features,dino_guidance_feat, prompt, gt_cls)

@SEM_SEG_HEADS_REGISTRY.register()
class FusionHeadVer11(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": FusionPredictorVer11(
                cfg,
            ),
        }

    def forward(self, features,dino_feat, guidance_features, prompt=None, gt_cls=None):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat,dino_feat, guidance_features, prompt, gt_cls)
    


@SEM_SEG_HEADS_REGISTRY.register()
class FusionHeadVer10(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": FusionPredictorVer10(
                cfg,
            ),
        }

    def forward(self, features,dino_feat, guidance_features, prompt=None, gt_cls=None):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat,dino_feat, guidance_features, prompt, gt_cls)
    
@SEM_SEG_HEADS_REGISTRY.register()
class FusionHeadVer09e(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": FusionPredictorVer09e(
                cfg,
            ),
        }

    def forward(self, features,dino_feat, guidance_features, dino_guidance_feat, prompt=None, gt_cls=None):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat,dino_feat, guidance_features,dino_guidance_feat, prompt, gt_cls)
    
@SEM_SEG_HEADS_REGISTRY.register()
class FusionHeadVer09d(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": FusionPredictorVer09d(
                cfg,
            ),
        }

    def forward(self, features,dino_feat, guidance_features, dino_guidance_feat, prompt=None, gt_cls=None):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat,dino_feat, guidance_features,dino_guidance_feat, prompt, gt_cls)
    
@SEM_SEG_HEADS_REGISTRY.register()
class FusionHeadVer09c(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": FusionPredictorVer09c(
                cfg,
            ),
        }

    def forward(self, features,dino_feat, guidance_features, dino_guidance_feat, prompt=None, gt_cls=None):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat,dino_feat, guidance_features,dino_guidance_feat, prompt, gt_cls)
    
@SEM_SEG_HEADS_REGISTRY.register()
class FusionHeadVer09b(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": FusionPredictorVer09b(
                cfg,
            ),
        }

    def forward(self, features,dino_feat, guidance_features, prompt=None, gt_cls=None):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat,dino_feat, guidance_features, prompt, gt_cls)
    
@SEM_SEG_HEADS_REGISTRY.register()
class FusionHeadVer09a(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": FusionPredictorVer09a(
                cfg,
            ),
        }

    def forward(self, features,dino_feat, guidance_features, prompt=None, gt_cls=None):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat,dino_feat, guidance_features, prompt, gt_cls)
    
@SEM_SEG_HEADS_REGISTRY.register()
class FusionHeadVer09(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": FusionPredictorVer09(
                cfg,
            ),
        }

    def forward(self, features,dino_feat, guidance_features, prompt=None, gt_cls=None):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat,dino_feat, guidance_features, prompt, gt_cls)
    
@SEM_SEG_HEADS_REGISTRY.register()
class FusionHeadVer08(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": FusionPredictorVer08(
                cfg,
            ),
        }

    def forward(self, features,dino_feat, guidance_features, prompt=None, gt_cls=None):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat,dino_feat, guidance_features, prompt, gt_cls)


@SEM_SEG_HEADS_REGISTRY.register()
class FusionHeadVer07(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": FusionPredictorVer07(
                cfg,
            ),
        }

    def forward(self, features,dino_feat, guidance_features, prompt=None, gt_cls=None):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat,dino_feat, guidance_features, prompt, gt_cls)











@SEM_SEG_HEADS_REGISTRY.register()
class CATSegHead(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": CATSegPredictor(
                cfg,
            ),
        }

    def forward(self, features, guidance_features, prompt=None, gt_cls=None):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat, guidance_features, prompt, gt_cls)