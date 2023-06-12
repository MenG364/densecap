import argparse
import os
import json

import torch
import numpy as np
from torch.utils.data.dataset import Subset
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torch.utils.tensorboard import SummaryWriter

from model.evaluator import DenseCapEvaluator
from utils.data_loader import DenseCapDataset, DataLoaderPFG
from model.densecap import densecap_resnet50_fpn

from evaluate import quality_check, quantity_check
from tqdm import tqdm
from describe import load_model, save_results_to_file

torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 10
USE_TB = True
CONFIG_PATH = './model_params'
MODEL_NAME = 'test_all_val_all_bz_2_epoch_10_inject_init_2'
IMG_DIR_ROOT = './data/visual-genome'
VG_DATA_PATH = './data/VG-regions-lite.h5'
LOOK_UP_TABLES_PATH = './data/VG-regions-dicts-lite.pkl'
MAX_TRAIN_IMAGE = -1  # if -1, use all images in train set
MAX_VAL_IMAGE = -1
os.environ['TORCH_HOME'] = 'E:\pycharmprojects\densecap-pytorch-main'


def set_args():
    args = dict()

    args['backbone_pretrained'] = True
    args['return_features'] = False

    # Caption parameters
    args['feat_size'] = 4096
    args['hidden_size'] = 512
    args['max_len'] = 16
    args['emb_size'] = 512
    args['rnn_num_layers'] = 1
    args['vocab_size'] = 10629
    args['fusion_type'] = 'init_inject'

    # Training Settings
    args['detect_loss_weight'] = 1.
    args['caption_loss_weight'] = 1.
    args['lr'] = 1e-4
    args['caption_lr'] = 1e-3
    args['weight_decay'] = 0.
    args['batch_size'] = 4
    args['use_pretrain_fasterrcnn'] = True
    args['box_detections_per_img'] = 50

    if not os.path.exists(os.path.join(CONFIG_PATH, MODEL_NAME)):
        os.mkdir(os.path.join(CONFIG_PATH, MODEL_NAME))
    with open(os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'), 'w') as f:
        json.dump(args, f, indent=2)

    return args


def quantity_check(model, dataset, idx_to_token, device, max_iter=-1, verbose=True):
    model.to(device)
    data_loader = DataLoaderPFG(dataset, batch_size=4, shuffle=False, num_workers=0,
                                pin_memory=True, collate_fn=DenseCapDataset.collate_fn)

    evaluator = DenseCapEvaluator(list(model.roi_heads.box_describer.special_idx.keys()))
    all_results=[]
    print('[quantity check]')
    try:
        t = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, (img, targets, info) in enumerate(data_loader):

            img = [img_tensor.to(device) for img_tensor in img]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            with torch.no_grad():
                model.eval()
                model.return_features = False
                detections = model(img)
                # all_results.extend([{k: v.cpu() for k, v in r.items()} for r in detections])

            for j in range(len(targets)):
                scores = detections[j]['scores']
                boxes = detections[j]['boxes']
                text = [' '.join(idx_to_token[idx] for idx in cap.tolist() if idx_to_token[idx] != '<pad>')
                        for cap in detections[j]['caps']]
                target_boxes = targets[j]['boxes']
                target_text = [' '.join(idx_to_token[idx] for idx in cap.tolist() if idx_to_token[idx] != '<pad>')
                               for cap in targets[j]['caps']]
                img_id = info[j]['file_name']

                evaluator.add_result(scores, boxes, text, target_boxes, target_text, img_id)

            if i >= max_iter > 0:
                break
            t.update(1)
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()

    results = evaluator.evaluate(verbose)
    if verbose:
        print('MAP: {:.3f} DET_MAP: {:.3f} METEOR: {:.3f}'.format(results['map'], results['detmap'], results['meteor']))

    return results, all_results


def test(args):
    print('Model {} start training...'.format(MODEL_NAME))

    model = load_model(args)

    model.to(device)

    test_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='test')
    idx_to_token = test_set.look_up_tables['idx_to_token']

    result, all_results = quantity_check(model, test_set, idx_to_token, device, max_iter=-1, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do dense captioning')
    parser.add_argument('--config_json', type=str, help="path of the json file which stored model configuration",
                        default="E:\PycharmProjects\densecap-pytorch-main\model_params\\train_all_val_all_bz_2_epoch_10_inject_init\config.json")
    parser.add_argument('--lut_path', type=str, default='./data/VG-regions-dicts-lite.pkl', help='look up table path')
    parser.add_argument('--model_checkpoint', type=str, help="path of the trained model checkpoint",
                        default="E:\PycharmProjects\densecap-pytorch-main\model_params\\train_all_val_all_bz_2_epoch_10_inject_init\\train_all_val_all_bz_2_epoch_10_inject_init.pth.tar")
    parser.add_argument('--img_path', type=str, help="path of images, should be a file or a directory with only images",
                        default="E:\PycharmProjects\densecap-pytorch-main\data\\visual-genome\VG_100K")
    parser.add_argument('--result_dir', type=str, default='./output',
                        help="path of the directory to save the output file")
    parser.add_argument('--box_per_img', type=int, default=100, help='max boxes to describe per image')
    parser.add_argument('--batch_size', type=int, default=1, help="useful when img_path is a directory")
    parser.add_argument('--extract', action='store_true', help='whether to extract features')
    parser.add_argument('--cpu', action='store_true', help='whether use cpu to compute')
    parser.add_argument('--verbose', action='store_true', help='whether output info')
    parser.add_argument('--check', action='store_true', help='whether to validate box feat by regenerate sentences')
    args = parser.parse_args()

    test(args)
