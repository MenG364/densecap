import torch
import torch.nn.functional as F
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign, RoIAlign

from model.attr_roiheads import DenseCapRoIHeads
from model.box_describer import BoxDescriber
# from model.roi_heads import DenseCapRoIHeads

__all__ = [
    "DenseCapModel", "densecap_resnet50_fpn",
]

from mydense import get_backbone


class DenseCapModel(GeneralizedRCNN):

    def __init__(self, attr, return_features=False,
                 # Caption parameters
                 box_describer=None,
                 feat_size=None, hidden_size=None, max_len=None,
                 emb_size=None, rnn_num_layers=None, vocab_size=None,
                 fusion_type='init_inject',
                 # transform parameters
                 min_size=300, max_size=720,  # 300²»È·¶¨
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.5, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(attr.backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if box_describer is None:
            for param in {max_len, emb_size, rnn_num_layers, vocab_size}:
                assert isinstance(param, int) and param > 0, 'invalid parameters of caption'
        else:
            assert max_len is None and emb_size is None and rnn_num_layers is None and vocab_size is None

        out_channels = attr.backbone.out_channels
        rpn=attr.rpn
        # for param in attr.backbone.named_parameters():
        #     param[1].requires_grad = False
        # for param in rpn.named_parameters():
        #     param[1].requires_grad = False
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)
            # box_roi_pool = RoIAlign(
            #     output_size=7,
            #     spatial_scale=1/16,
            #     sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            # resolution = box_roi_pool.output_size
            representation_size = 4096 if feat_size is None else feat_size
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 4096 if feat_size is None else feat_size
            box_predictor = FastRCNNPredictor(
                representation_size,
                1595)

        if box_describer is None:
            representation_size = 4096 if feat_size is None else feat_size
            box_describer = BoxDescriber(representation_size, hidden_size, max_len,
                                         emb_size, rnn_num_layers, vocab_size, fusion_type)


        roi_heads = DenseCapRoIHeads(
            # Caption
            box_describer,
            attr.roi_heads,
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            # Whether return features during testing
            return_features)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(DenseCapModel, self).__init__(attr.backbone, rpn, roi_heads, transform)


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def densecap_resnet50_fpn(attr,backbone_pretrained=False, **kwargs):

    model = DenseCapModel(attr,**kwargs)

    return model

def densecap_resnet101_fpn(backbone_pretrained=False, **kwargs):


    backbone = resnet_fpn_backbone('resnet101', backbone_pretrained)
    model = DenseCapModel(backbone,**kwargs)

    return model


def fasterrcnn_resnet101_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet101', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = torch.load('hub/checkpoints/e2e_faster_rcnn_R_101_FPN_1x.pth')
        model.load_state_dict(state_dict)
    return model

def fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    attr, _, _ = get_backbone()
    if pretrained:
        state_dict = torch.load('hub/checkpoints/e2e_faster_rcnn_R_50_FPN_1x.pth')
        state_dict = state_dict['model']
        model.load_state_dict(state_dict['model'])
    return model

