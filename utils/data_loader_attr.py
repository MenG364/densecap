import json
import math
import os
import pickle

import cv2
import h5py
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

__all__ = [
    "DataLoaderPFG", "AttrDenseCapDataset"
]

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '') \
            .replace('?', '').replace('\'s', ' \'s').replace('.', '')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK
                # for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx - 1))
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def cv2Img_to_Image(input_img):
    cv2_img = input_img.copy()
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


class DataLoaderPFG(DataLoader):
    """
    Prefetch version of DataLoader: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class ClevrDenseCapDataset(Dataset):
    """Images are loaded from by open specific file
    """

    @staticmethod
    def collate_fn(batch):
        """Use in torch.utils.data.DataLoader
        """

        return tuple(zip(*batch))  # as tuples instead of stacked tensors

    @staticmethod
    def get_transform():
        """More complicated transform utils in torchvison/references/detection/transforms.py
        """

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return transform

    def __init__(self,img_dir_root,dictionary, dataset_type=None, transform=None):

        assert dataset_type in {None, 'train', 'test', 'val'}

        super(ClevrDenseCapDataset, self).__init__()

        self.img_dir_root = img_dir_root
        self.dataset_type = dataset_type  # if dataset_type is None, all data will be use
        self.bbox = json.load(open(os.path.join(self.img_dir_root, 'features', '%s_scenes_with_bb.json' % self.dataset_type),'rb'))
        self.label2ix = self.bbox['label_to_ix']
        self.ix2label = self.bbox['ix_to_label']
        self.annotations = self.bbox['annotations']
        self.transform = transform
        self.dictionary=dictionary

        for anno in self.annotations:
            bbox = []
            label = []
            for obj in anno['objects']:
                bbox.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
                label.append(obj['label_id'])
            anno['bbox'] = bbox
            anno['label'] = label

    def set_dataset_type(self, dataset_type, verbose=True):

        assert dataset_type in {None, 'train', 'test', 'val'}

        if verbose:
            print('[DenseCapDataset]: {} switch to {}'.format(self.dataset_type, dataset_type))

        self.dataset_type = dataset_type

    def __getitem__(self, idx):
        annotations = self.annotations[idx]
        image_id = annotations['image_id']
        filename = annotations['filename']

        boxes = torch.as_tensor(annotations['bbox'])
        clss =  torch.as_tensor(annotations['label'])
        img_path = os.path.join(self.img_dir_root, 'images', self.dataset_type, filename)
        cv2_img = cv2.imread(img_path)
        img = cv2Img_to_Image(cv2_img)
        img_size = img.size
        targets = {
            'boxes': boxes,
            'class': clss
        }
        (image_width, image_height) = img_size
        # anchors_in_image = []
        boxlist = BoxList(
            targets['boxes'], (image_width, image_height), mode="xyxy"
        )
        # anchors_in_image.append(boxlist)
        boxlist.add_field('labels', targets['class'])

        if self.transform is not None:
            img, target = self.transform(img, boxlist)
        else:
            img = transforms.ToTensor()(img)
        # img=self.img[idx]
        new_img_size = img.shape[1:]
        scale = math.sqrt(float(new_img_size[0] * new_img_size[1]) / float(img_size[0] * img_size[1]))

        info = {
            'idx': image_id,
            'dir': img_path,
            'file_name': filename
        }

        return img, target, info, scale

    def __len__(self):
        return len(self.annotations)


class AttrDenseCapDataset(Dataset):
    """Images are loaded from by open specific file
    """

    @staticmethod
    def collate_fn(batch):
        """Use in torch.utils.data.DataLoader
        """

        return tuple(zip(*batch))  # as tuples instead of stacked tensors

    @staticmethod
    def get_transform():
        """More complicated transform utils in torchvison/references/detection/transforms.py
        """

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return transform

    def __init__(self, rgid2cls, img_dir_root, vg_data_path, look_up_tables_path, dataset_type=None, transform=None):

        assert dataset_type in {None, 'train', 'test', 'val'}

        super(AttrDenseCapDataset, self).__init__()

        self.img_dir_root = img_dir_root
        self.vg_data_path = vg_data_path
        self.look_up_tables_path = look_up_tables_path
        self.dataset_type = dataset_type  # if dataset_type is None, all data will be use
        self.transform = transform
        self.labelmap = json.load(open('data/VG-SGG-dicts-vgoi6-clipped.json', 'rb'))
        self.regionid2class = rgid2cls

        # === load data here ====
        self.look_up_tables = pickle.load(open(look_up_tables_path, 'rb'))
        # self.img=[]
        # with h5py.File(self.vg_data_path, 'r') as vg_data:
        #     for idx in range(len(self.look_up_tables['split'][self.dataset_type])):
        #         vg_idx = self.look_up_tables['split'][self.dataset_type][idx] if self.dataset_type else idx
        #
        #         img_path = os.path.join(self.img_dir_root, self.look_up_tables['idx_to_directory'][vg_idx],
        #                                 self.look_up_tables['idx_to_filename'][vg_idx])
        #
        #         # img = Image.open(img_path).convert("RGB")
        #         cv2_img = cv2.imread(img_path)
        #         img = cv2Img_to_Image(cv2_img)
        #         img_size = img.size
        #
        #         if self.transform is not None:
        #             img, _ = self.transform(img, target=None)
        #         else:
        #             img = transforms.ToTensor()(img)
        #         self.img.append(img)

    def set_dataset_type(self, dataset_type, verbose=True):

        assert dataset_type in {None, 'train', 'test', 'val'}

        if verbose:
            print('[DenseCapDataset]: {} switch to {}'.format(self.dataset_type, dataset_type))

        self.dataset_type = dataset_type

    def __getitem__(self, idx):

        with h5py.File(self.vg_data_path, 'r') as vg_data:

            vg_idx = self.look_up_tables['split'][self.dataset_type][idx] if self.dataset_type else idx

            img_path = os.path.join(self.img_dir_root, self.look_up_tables['idx_to_directory'][vg_idx],
                                    self.look_up_tables['idx_to_filename'][vg_idx])
            first_box_idx = vg_data['img_to_first_box'][vg_idx]
            last_box_idx = vg_data['img_to_last_box'][vg_idx]

            boxes = torch.as_tensor(vg_data['boxes'][first_box_idx: last_box_idx + 1], dtype=torch.float32)
            caps = torch.as_tensor(vg_data['captions'][first_box_idx: last_box_idx + 1], dtype=torch.long)
            caps_len = torch.as_tensor(vg_data['lengths'][first_box_idx: last_box_idx + 1], dtype=torch.long)

            imgid = int(self.look_up_tables['idx_to_filename'][vg_idx].split('.')[0])
            clss = self.regionid2class[imgid]
            clss = torch.as_tensor([self.labelmap['label_to_idx'][v] for v in clss.values()])

            # img = Image.open(img_path).convert("RGB")
            cv2_img = cv2.imread(img_path)
            img = cv2Img_to_Image(cv2_img)
            img_size = img.size
            targets = {
                'boxes': boxes,
                'caps': caps,
                'caps_len': caps_len,
                'class': clss
            }
            (image_width, image_height) = img_size
            # anchors_in_image = []
            boxlist = BoxList(
                targets['boxes'], (image_width, image_height), mode="xyxy"
            )
            # anchors_in_image.append(boxlist)
            boxlist.add_field('labels', targets['class'])
            boxlist.add_field('caps', targets['caps'])
            boxlist.add_field('caps_len', targets['caps_len'])

            if self.transform is not None:
                img, target = self.transform(img, boxlist)
            else:
                img = transforms.ToTensor()(img)
            # img=self.img[idx]
            new_img_size = img.shape[1:]
            scale = math.sqrt(float(new_img_size[0] * new_img_size[1]) / float(img_size[0] * img_size[1]))

            info = {
                'idx': vg_idx,
                'dir': self.look_up_tables['idx_to_directory'][vg_idx],
                'file_name': self.look_up_tables['idx_to_filename'][vg_idx]
            }

        return img, target, info, scale

    def __len__(self):

        if self.dataset_type:
            return len(self.look_up_tables['split'][self.dataset_type])
        else:
            return len(self.look_up_tables['filename_to_idx'])
