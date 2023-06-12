# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import os
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from maskrcnn_benchmark.structures.bounding_box import BoxList

from maskrcnn_benchmark.modeling.roi_heads.mask_head.mask_head import keep_only_positive_boxes
from model.box_describer import BoxDescriber
import torch
import torch.nn.functional as F

class ROICaptionHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROICaptionHead, self).__init__()
        self.cfg = cfg.clone()
        self.box_describer = BoxDescriber(in_channels, 512, 17,
                                         512, 1, 8776, 'init_inject')
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, box_features, proposals, targets=None):

        labels = None
        matched_idxs = None
        caption_gt = None
        caption_length = None
        regression_targets = None
        if self.training:
            box_features = self.avgpool(box_features)
            box_features = box_features.view(box_features.size(0), -1)
            labels=[box.extra_fields["labels"] for box in proposals]
            caption_gt=[box.extra_fields["caps"] for box in proposals]
            caption_length=[box.extra_fields["caps_len"] for box in proposals]

            keep_ids = [label > 0 for label in labels]
            boxes_per_image = [boxes_in_image.bbox.shape[0] for boxes_in_image in proposals]
            box_features = box_features.split(boxes_per_image, 0)
            box_features_gt = []
            for i in range(len(keep_ids)):
                box_features_gt.append(box_features[i][keep_ids[i]])
                caption_gt[i] = caption_gt[i][keep_ids[i]]
                caption_length[i] = caption_length[i][keep_ids[i]]
            box_features = torch.cat(box_features_gt, 0)
        else:
            box_features = torch.cat([p.extra_fields['box_features'] for p in proposals],0)
        caption_predicts = self.box_describer(box_features, caption_gt, caption_length)

        if not self.training:

            result = self.post_processor(caption_predicts, proposals, box_features)
            return result, {}
        if self.training:
            loss_caption = caption_loss(caption_predicts, caption_gt, caption_length)
            return proposals, dict(loss_caption=loss_caption)

    def post_processor(self, x, boxes, features):
        boxes_per_image = [len(box) for box in boxes]
        pred_caption = x.split(boxes_per_image, 0)
        results = []
        for box, pred in zip(boxes, pred_caption):
            # copy the current boxes
            boxlist = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                boxlist.add_field(field, box.get_field(field))
            boxlist.add_field('caps', pred)
            results.append(boxlist)

        return results

def caption_loss(caption_predicts, caption_gt, caption_length):
    """
    Computes the loss for caption part.
    Arguments:
        caption_predicts (Tensor)
        caption_gt (Tensor or list[Tensor])
        caption_length (Tensor or list[Tensor])
        caption_loss (Tensor)
    """

    if isinstance(caption_gt, list) and isinstance(caption_length, list):
        caption_gt = torch.cat(caption_gt, dim=0)  # (batch_size, max_len+1)
        caption_length = torch.cat(caption_length, dim=0)  # (batch_size, )
        assert caption_predicts.shape[0] == caption_gt.shape[0] and caption_predicts.shape[0] == caption_length.shape[0]

    # '<bos>' is not considered
    caption_length = torch.clamp(caption_length - 1, min=0)

    predict_pps = pack_padded_sequence(caption_predicts, caption_length.cpu(), batch_first=True, enforce_sorted=False)

    target_pps = pack_padded_sequence(caption_gt[:, 1:], caption_length.cpu(), batch_first=True, enforce_sorted=False)

    return F.cross_entropy(predict_pps.data, target_pps.data)
