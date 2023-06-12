import argparse
import os

import torch
import torch.nn.functional as F
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from scene_graph_benchmark.config import sg_cfg
from torch import nn
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.collate_batch import BBoxAugCollator, BatchCollator
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from model.box_describer import BoxDescriber
from model.roi_heads import caption_loss


def get_backbone():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="sgg_configs/vgattr/vinvl_x152c4.yaml",
        # default="sgg_configs/e2e_faster_rcnn_R_101_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--extract', action='store_true', help='whether to extract features')

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TEST.IMS_PER_BATCH = 2
    cfg.MODEL.WEIGHT = 'models/vinvl/vinvl_vg_x152c4.pth'
    # cfg.MODEL.WEIGHT = 'hub/checkpoints/e2e_faster_rcnn_R_101_FPN_1x.pth'
    cfg.MODEL.ROI_HEADS.NMS_FILTER = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH = 0.49
    cfg.MODEL.ROI_HEADS.NMS = 0.3
    cfg.TEST.IGNORE_BOX_REGRESSION = False
    cfg.TEST.OUTPUT_FEATURE = True
    cfg.freeze()

    model = AttrRCNN(cfg)
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
    transforms_train = build_transforms(cfg, True)
    transforms_val = build_transforms(cfg, False)

    collator = BBoxAugCollator() if not True and cfg.TEST.BBOX_AUG.ENABLED else \
        BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
    # for param in model.named_parameters():
    #     param[1].requires_grad = False
    return model, transforms_train, transforms_val,collator


class MyDenseCap(nn.Module):
    def __init__(self, backbone_model,feat_size=None, hidden_size=None, max_len=None,
                 emb_size=None, rnn_num_layers=None, vocab_size=None, fusion_type='init_inject', ):
        super(MyDenseCap, self).__init__()
        self.backbone = backbone_model.backbone
        representation_size = 2048
        box_describer = BoxDescriber(representation_size, hidden_size, max_len,
                                     emb_size, rnn_num_layers, vocab_size, fusion_type)
        self.roi_heads=backbone_model.roi_heads
        self.rpn=backbone_model.rpn

        self.proposal_matcher = det_utils.Matcher(
            0.5,
            0.5,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            512,
            0.25)


        bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)
        self.box_describer = box_describer

        self.score_thresh = 0.5
        self.nms_thresh = 0.5
        self.detections_per_img = 100
        # for param in self.backbone.named_parameters():
        #     param[1].requires_grad = False
        # for param in self.roi_heads.named_parameters():
        #     param[1].requires_grad = False


    def forward(self, images, target=None):
        self.backbone.eval()
        self.roi_heads.eval()
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        anchors_over_all_feature_maps = [b['boxes'] for b in target]
        bbox = []
        for i, (image_height, image_width) in enumerate(images.image_sizes):
            # anchors_in_image = []
            boxlist = BoxList(
                anchors_over_all_feature_maps[i], (image_width, image_height), mode="xyxy"
            )
            # anchors_in_image.append(boxlist)
            bbox.append(boxlist)
        _, predictions, detector_losses = self.roi_heads(features,
                                                         bbox, [])
        proposals=[p.bbox for p in predictions]
        if self.training:
            proposals, matched_idxs, caption_gt, caption_length, labels, regression_targets = \
                self.select_training_samples(proposals, target)  # [512 512]
        else:
            labels = None
            matched_idxs = None
            caption_gt = None
            caption_length = None
            regression_targets = None
        y = [p.extra_fields['box_features'] for p in predictions]
        box_features = torch.cat(y, 0)
        if self.training:
            # labels 到这里应该是有0和（1，class-1），0代表背景，其余代表类别，需要剔除背景，然后进行描述(List[Tensor])
            # 也需要滤除对应的caption和caption_length
            keep_ids = [label > 0 for label in labels]
            boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
            box_features = box_features.split(boxes_per_image, 0)
            box_features_gt = []
            for i in range(len(keep_ids)):
                box_features_gt.append(box_features[i][keep_ids[i]])
                caption_gt[i] = caption_gt[i][keep_ids[i]]
                caption_length[i] = caption_length[i][keep_ids[i]]
            box_features = torch.cat(box_features_gt, 0)

        # y = [torch.index_select(p,0,matched_idxs[i]) for i,p in enumerate(y)]

        caption_predicts = self.box_describer(box_features, caption_gt, caption_length)

        result, losses = [], {}
        if self.training:
            # loss_classifier, loss_box_reg = detect_loss(logits, box_regression, labels, regression_targets)
            loss_caption = caption_loss(caption_predicts, caption_gt, caption_length)

            losses = {
                "loss_caption": loss_caption
            }
        else:
            boxes, scores, caption_predicts, feats = self.postprocess_detections(logits, box_regression,
                                                                                 caption_predicts, proposals,
                                                                                 image_shapes, box_features,
                                                                                 self.return_features)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "caps": caption_predicts[i],
                        "scores": scores[i],
                    }
                )
                if self.return_features:
                    result[-1]['feats'] = feats[i]

        return result, losses

    def select_training_samples(self, proposals, targets):
        """
        proposals: (List[Tensor[N, 4]])
        targets (List[Dict])
        """
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_captions = [t["caps"] for t in targets]
        gt_captions_length = [t["caps_len"] for t in targets]
        gt_labels = [torch.ones((t["boxes"].shape[0],), dtype=torch.int64, device=device) for t in
                     targets]  # generate labels LongTensor(1)

        # append ground-truth bboxes to propos
        # List[2*N,4],一个list是一张图片
        # proposals = [
        #     torch.cat((proposal, gt_box))
        #     for proposal, gt_box in zip(proposals, gt_boxes)
        # ]

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]  # (M,) 0~P-1
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]  # before (P,) / after (M,) 0~N-1

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
            gt_captions[img_id] = gt_captions[img_id][matched_idxs[img_id]]  # before (N, ) / after (M, )
            gt_captions_length[img_id] = gt_captions_length[img_id][matched_idxs[img_id]]

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        return proposals, matched_idxs, gt_captions, gt_captions_length, labels, regression_targets
    def postprocess_detections(self, logits, box_regression, caption_predicts, proposals, image_shapes,
                               box_features, return_features):
        device = logits.device
        num_classes = logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_caption_list = caption_predicts.split(boxes_per_image, 0)
        if return_features:
            pred_box_features_list = box_features.split(boxes_per_image, 0)
        else:
            pred_box_features_list = None

        all_boxes = []
        all_scores = []
        all_labels = []
        all_captions = []
        all_box_features = []
        remove_inds_list = []
        keep_list = []
        for boxes, scores, captions, image_shape in zip(pred_boxes_list, pred_scores_list, pred_caption_list,
                                                        image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            remove_inds_list.append(inds)
            boxes, scores, captions, labels = boxes[inds], scores[inds], captions[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, captions, labels = boxes[keep], scores[keep], captions[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            keep_list.append(keep)
            boxes, scores, captions, labels = boxes[keep], scores[keep], captions[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_captions.append(captions)
            all_labels.append(labels)

        if return_features:
            for inds, keep, box_features in zip(remove_inds_list, keep_list, pred_box_features_list):
                all_box_features.append(box_features[inds[keep] // (num_classes - 1)])

        return all_boxes, all_scores, all_captions, all_box_features
    def subsample(self, labels):

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds
    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):

        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):  # 每张图片循环

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                # iou (Tensor[N, M]): the NxM matrix containing the IoU values for every element in boxes1 and boxes2

                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = torch.tensor(0).cuda()

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = torch.tensor(-1).cuda()  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels