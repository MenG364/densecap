# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import os
import torch

from maskrcnn_benchmark.modeling.roi_heads.mask_head.mask_head import keep_only_positive_boxes

from .roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from .roi_attribute_predictors import make_roi_attribute_predictor
from .inference import make_roi_attribute_post_processor
from .loss import make_roi_attribute_loss_evaluator

class ROIAttributeHeadAddAttr(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIAttributeHeadAddAttr, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_attribute_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_attribute_post_processor(cfg)
        self.loss_evaluator = make_roi_attribute_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from box_head
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the attribute feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `attribute` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)


        labels = torch.cat(
                [boxes_per_image.get_field("labels").view(-1) for boxes_per_image in proposals], dim=0)
        attribute_logits, attribute_features = self.predictor(features, labels)

        if not self.training:
            result = self.post_processor(attribute_logits, proposals, attribute_features)
            return features, result, {}



def build_roi_attribute_head(cfg, in_channels):
    return ROIAttributeHeadAddAttr(cfg, in_channels)
