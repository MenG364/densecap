import os

import cv2
import h5py
import json
import pickle
import argparse

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
# import torchvision.transforms as transforms

from model.densecap import densecap_resnet50_fpn

from mydense import get_backbone
from paint import plot_boxes
from utils.data_loader_attr import cv2Img_to_Image
from torch.utils.data import Dataset, DataLoader

os.environ['TORCH_HOME'] = 'E:\PycharmProjects\densecap-pytorch-main'

# os.environ['TORCH_HOME'] = '/media/lrg/文档/PycharmProjects/densecap'
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class ImgDataset(Dataset):
    @staticmethod
    def collate_fn(batch):
        """Use in torch.utils.data.DataLoader
        """

        return tuple(zip(*batch))  # as tuples instead of stacked tensors

    def __init__(self, img_list, transforms):
        super(ImgDataset, self).__init__()
        self.img_list = img_list
        self.transforms = transforms

    def __getitem__(self, idx):
        img = self.img_to_tensor([self.img_list[idx]], self.transforms)
        return img

    def __len__(self):
        return len(self.img_list)

    def img_to_tensor(self, img_list, transforms):
        assert isinstance(img_list, list) and len(img_list) > 0

        img_tensors = []

        for img_path in img_list:
            # img = Image.open(img_path).convert("RGB")
            # img_tensors.append(transforms.ToTensor()(img_input))
            cv2_img = cv2.imread(img_path)
            img_input = cv2Img_to_Image(cv2_img)
            img_input, _ = transforms(img_input, target=None)
            img_height = cv2_img.shape[0]
            img_width = cv2_img.shape[1]

            img_tensors.append([img_input, img_height, img_width])

        return img_tensors


def load_model(console_args):
    with open(console_args.config_json, 'r') as f:
        model_args = json.load(f)

    model = densecap_resnet50_fpn(backbone_pretrained=model_args['backbone_pretrained'],
                                  return_features=console_args.extract,
                                  feat_size=model_args['feat_size'],
                                  hidden_size=model_args['hidden_size'],
                                  max_len=model_args['max_len'],
                                  emb_size=model_args['emb_size'],
                                  rnn_num_layers=model_args['rnn_num_layers'],
                                  vocab_size=model_args['vocab_size'],
                                  fusion_type=model_args['fusion_type'],
                                  box_detections_per_img=console_args.box_per_img,
                                  box_nms_thresh=0.5,
                                  box_score_thresh=0.4)

    checkpoint = torch.load(console_args.model_checkpoint)
    model.load_state_dict(checkpoint['model'])

    if console_args.verbose and 'results_on_val' in checkpoint.keys():
        print('[INFO]: checkpoint {} loaded'.format(console_args.model_checkpoint))
        print('[INFO]: correspond performance on val set:')
        for k, v in checkpoint['results_on_val'].items():
            if not isinstance(v, dict):
                print('        {}: {:.3f}'.format(k, v))

    return model


def load_model1(console_args):
    with open(console_args.config_json, 'r') as f:
        args = json.load(f)

    attr, transforms, collator = get_backbone(is_train=False)
    from model.attr_dense import densecap_resnet50_fpn
    model = densecap_resnet50_fpn(attr, backbone_pretrained=args['backbone_pretrained'],
                                  feat_size=args['feat_size'],
                                  hidden_size=args['hidden_size'],
                                  max_len=args['max_len'],
                                  emb_size=args['emb_size'],
                                  rnn_num_layers=args['rnn_num_layers'],
                                  vocab_size=args['vocab_size'],
                                  fusion_type=args['fusion_type'],
                                  box_detections_per_img=args['box_detections_per_img'],
                                  box_nms_thresh=0.5,
                                  box_score_thresh=0.5)

    checkpoint = torch.load(console_args.model_checkpoint)
    model.load_state_dict(checkpoint['model'])

    if console_args.verbose and 'results_on_val' in checkpoint.keys():
        print('[INFO]: checkpoint {} loaded'.format(console_args.model_checkpoint))
        print('[INFO]: correspond performance on val set:')
        for k, v in checkpoint['results_on_val'].items():
            if not isinstance(v, dict):
                print('        {}: {:.3f}'.format(k, v))

    return model


def load_model2(console_args):
    model, transforms_trian, transforms_val, collator = get_backbone()
    checkpoint = torch.load(console_args.model_checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)

    return model, transforms_val


def get_image_path(console_args):
    img_list = []

    if os.path.isdir(console_args.img_path):
        for file_name in os.listdir(console_args.img_path):
            img_list.append(os.path.join(console_args.img_path, file_name))
    else:
        img_list.append(console_args.img_path)

    return img_list


def img_to_tensor(img_list, transforms):
    assert isinstance(img_list, list) and len(img_list) > 0

    img_tensors = []

    for img_path in img_list:
        # img = Image.open(img_path).convert("RGB")
        # img_tensors.append(transforms.ToTensor()(img_input))
        cv2_img = cv2.imread(img_path)
        img_input = cv2Img_to_Image(cv2_img)
        img_input, _ = transforms(img_input, target=None)
        img_height = cv2_img.shape[0]
        img_width = cv2_img.shape[1]

        img_tensors.append([img_input, img_height, img_width])

    return img_tensors


def describe_images(model, img_list, device, console_args, transforms):
    assert isinstance(img_list, list)
    assert isinstance(console_args.batch_size, int) and console_args.batch_size > 0
    dataset = ImgDataset(img_list, transforms)
    dataloader = DataLoaderX(dataset, batch_size=console_args.batch_size, num_workers=0, pin_memory=True,
                             collate_fn=ImgDataset.collate_fn)

    all_results = []

    with torch.no_grad():
        model.to(device)
        model.eval()
        try:
            t = tqdm(range(0, len(img_list), console_args.batch_size))
            for i in t:
                image_tensors = img_to_tensor(img_list[i:i + console_args.batch_size], transforms)
                # t = tqdm(total=len(dataloader))
                # for i, image_tensors in enumerate(dataloader):
                #     image_tensors = image_tensors[0]
                input_ = [t[0].to(device) for t in image_tensors]

                results = model(input_)
                for i in range(len(results)):
                    img_height = image_tensors[i][1]
                    img_width = image_tensors[i][2]
                    results[i] = results[i].resize((img_width, img_height))
                all_results.extend([{'boxes': r.bbox.cpu(),
                                     'labels': r.extra_fields['labels'].cpu(),
                                     'scores': r.extra_fields['scores'].cpu(),
                                     'caps': r.extra_fields['caps'].cpu(),
                                     'feats': r.extra_fields['box_features'].cpu()} for r in results])
                # all_results.extend([{k: v.cpu() for k, v in r.items()} for r in results])
                t.update(1)
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()

    return all_results


def save_results_to_file(img_list, all_results, console_args):
    with open(os.path.join(console_args.lut_path), 'rb') as f:
        look_up_tables = pickle.load(f)

    idx_to_token = look_up_tables['idx_to_token']

    results_dict = {}
    if console_args.extract:
        total_box = sum(len(r['boxes']) for r in all_results)
        start_idx = 0
        img_idx = 0
        h = h5py.File(os.path.join(console_args.result_dir, 'box_feat.h5'), 'w')
        h.create_dataset('feats', (total_box, all_results[0]['feats'].shape[1]), dtype=np.float32)
        h.create_dataset('boxes', (total_box, 4), dtype=np.float32)
        h.create_dataset('start_idx', (len(img_list),), dtype=np.long)
        h.create_dataset('end_idx', (len(img_list),), dtype=np.long)

    for img_path, results in zip(img_list, all_results):

        if console_args.verbose:
            print('[Result] ==== {} ====='.format(img_path))

        results_dict[img_path] = []
        for box, cap, score in zip(results['boxes'], results['caps'], results['scores']):

            r = {
                'box': [round(c, 2) for c in box.tolist()],
                'score': round(score.item(), 2),
                'cap': ' '.join(idx_to_token[idx] for idx in cap.tolist()
                                if idx_to_token[idx] not in ['<pad>', '<bos>', '<eos>'])
            }

            if console_args.verbose and r['score'] > 0.9:
                print('        SCORE {}  BOX {}'.format(r['score'], r['box']))
                print('        CAP {}\n'.format(r['cap']))

            results_dict[img_path].append(r)

        if console_args.extract:
            box_num = len(results['boxes'])
            h['feats'][start_idx: start_idx + box_num] = results['feats'].cpu().numpy()
            h['boxes'][start_idx: start_idx + box_num] = results['boxes'].cpu().numpy()
            h['start_idx'][img_idx] = start_idx
            h['end_idx'][img_idx] = start_idx + box_num - 1
            start_idx += box_num
            img_idx += 1

    if console_args.extract:
        h.close()
        # save order of img to a txt
        if len(img_list) > 1:
            with open(os.path.join(console_args.result_dir, 'feat_img_mappings.txt'), 'w') as f:
                for img_path in img_list:
                    f.writelines(os.path.split(img_path)[1] + '\n')

    if not os.path.exists(console_args.result_dir):
        os.mkdir(console_args.result_dir)
    with open(os.path.join(console_args.result_dir, 'result.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)

    if console_args.verbose:
        print('[INFO] result save to {}'.format(os.path.join(console_args.result_dir, 'result.json')))
        if console_args.extract:
            print('[INFO] feats save to {}'.format(os.path.join(console_args.result_dir, 'box_feat.h5')))
            print('[INFO] order save to {}'.format(os.path.join(console_args.result_dir, 'feat_img_mappings.txt')))


def validate_box_feat(model, all_results, device, console_args):
    with torch.no_grad():

        box_describer = model.roi_heads.box_describer
        box_describer.to(device)
        box_describer.eval()

        if console_args.verbose:
            print('[INFO] start validating box features...')
        for results in tqdm(all_results, disable=not console_args.verbose):
            captions = box_describer(results['feats'].to(device))

            assert (captions.cpu() == results['caps']).all().item(), 'caption mismatch'

    if console_args.verbose:
        print('[INFO] validate box feat done, no problem')


def main(console_args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # === prepare images ====
    img_list = get_image_path(console_args)

    # === prepare model ====
    model, transforms = load_model2(console_args)

    # === inference ====
    all_results = describe_images(model, img_list, device, console_args, transforms)

    # === save results ====
    img = Image.open(img_list[0]).convert('RGB')
    # img = cv2.imread(img_list[0])
    bboxes = all_results[0]['boxes'].tolist()
    scores = all_results[0]['scores'].tolist()
    # labels = all_results[0]['labels'].tolist()
    caps = all_results[0]['caps'].tolist()
    labels = []
    with open(os.path.join(console_args.lut_path), 'rb') as f:
        look_up_tables = pickle.load(f)
    idx_to_token = look_up_tables['idx_to_token']

    for cap in caps:
        labels.append(' '.join(idx_to_token[idx] for idx in cap
                               if idx_to_token[idx] not in ['<pad>', '<bos>', '<eos>']))

    for bbox, score, id in zip(bboxes, scores, labels):
        bbox.extend([score, id])

    clses = json.load(open('F:\datasets\VQA\VQA2.0old\data\VG-SGG-dicts-vgoi6-clipped.json', 'rb'))
    class_names = [0]
    for cls in clses['label_to_idx'].keys():
        class_names.append(cls)
    plot_boxes(img, bboxes[:10], 'img_136.jpg', scores=scores, class_names=None)
    # draw_bb(img, bboxes, labels, scores)
    # cv2.imwrite('img_10.jpg', img)
    print(len(all_results[0]['boxes']))
    save_results_to_file(img_list, all_results, console_args)

    if console_args.extract and console_args.check:
        validate_box_feat(model, all_results, device, console_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do dense captioning')
    parser.add_argument('--config_json', type=str, help="path of the json file which stored model configuration",
                        default="model_params/train_all_val_all_bz_2_epoch_10_inject_init_caption/config.json")
    parser.add_argument('--lut_path', type=str, default='data/VG-regions-dicts-lite.pkl', help='look up table path')
    parser.add_argument('--model_checkpoint', type=str, help="path of the trained model checkpoint",
                        default="model_params/train_all_val_all_bz_2_epoch_10_inject_init_attr/train_all_val_all_bz_2_epoch_10_inject_init_attr_best.pth.tar")
    # parser.add_argument('--img_path', type=str, help="path of images, should be a file or a directory with only images",default='F:\datasets\VQA\VQA2.0old\data\\visual-genome\VG_100K_2/2413345.jpg')
    # parser.add_argument('--img_path', type=str, help="path of images, should be a file or a directory with only images",default='F:\datasets\VQA\VQA2.0old\GQA\images')
    # parser.add_argument('--img_path', type=str, help="path of images, should be a file or a directory with only images",default='data/visual-genome/VG_100K/0_s_0_g2.png')
    parser.add_argument('--img_path', type=str, help="path of images, should be a file or a directory with only images",
                        default='F:\datasets\\val2014\\COCO_val2014_000000000136.jpg')
    # parser.add_argument('--img_path', type=str, help="path of images, should be a file or a directory with only images",default='data/img00001.png')
    parser.add_argument('--result_dir', type=str, default='./output',
                        help="path of the directory to save the output file")
    parser.add_argument('--box_per_img', type=int, default=100, help='max boxes to describe per image')
    parser.add_argument('--batch_size', type=int, default=1, help="useful when img_path is a directory")
    parser.add_argument('--extract', action='store_true', help='whether to extract features')
    parser.add_argument('--cpu', action='store_true', help='whether use cpu to compute')
    parser.add_argument('--verbose', action='store_true', help='whether output info')
    parser.add_argument('--check', action='store_true', help='whether to validate box feat by regenerate sentences')
    args = parser.parse_args()

    main(args)
