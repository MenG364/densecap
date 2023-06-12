import json
import math
import os
import pickle

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from scene_graph_benchmark.caption_head.evaluate import quantity_check
from mydense import get_backbone
from utils.data_loader_attr import AttrDenseCapDataset, DataLoaderPFG, ClevrDenseCapDataset, Dictionary

# sys.stdout = Logger(sys.stdout)
# sys.stderr = Logger(sys.stderr)

torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VG = pickle.load(open(LOOK_UP_TABLES_PATH, 'rb'))\
MAX_EPOCHS = 100
USE_TB = True
CONFIG_PATH = './model_params'
MODEL_NAME = 'train_clevr'
IMG_DIR_ROOT = 'F:\datasets\VQA\CLEVR'
MAX_TRAIN_IMAGE = -1  # if -1, use all images in train set
MAX_VAL_IMAGE = -1
# os.environ['TORCH_HOME'] = 'E:\pycharmprojects\densecap-pytorch-main'
os.environ['TORCH_HOME'] = 'E:\PycharmProjects\densecap-pytorch-main'


class lr_schedule_cosine():
    def __init__(self, lr_warm_up_schedule, T_0, T_mult=1, eta_max=1., eta_min=0., last_epoch=-1, restart=True):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.T_cur = last_epoch
        self.lr_warm_up_schedule = lr_warm_up_schedule
        self.warm_up_step = len(self.lr_warm_up_schedule)
        self.restart = restart
        self.last_epoch = last_epoch

    def compute_restart(self, epoch):
        if epoch != self.last_epoch:
            self.T_cur += 1
        # if self.T_cur == self.T_i and epoch != self.last_epoch:
        #     lr = self.eta_min
        #     self.T_i *= self.T_mult
        #     self.T_cur = -1
        if self.T_cur == 0:
            lr = self.eta_max
        else:
            lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        return lr

    def compute(self, epoch):
        if epoch != self.last_epoch:
            self.T_cur += 1
        if self.T_cur == 0 and epoch != self.last_epoch:
            lr = self.eta_max
        else:
            lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        return lr

    def get_lr(self, epoch):
        if epoch < len(self.lr_warm_up_schedule):
            lr_select = self.lr_warm_up_schedule[epoch]
        else:
            if self.restart:
                lr_select = self.compute_restart(epoch)
            else:
                lr_select = self.compute()
        self.last_epoch = epoch
        return lr_select


def set_args():
    args = dict()

    args['backbone_pretrained'] = True
    args['return_features'] = False

    # Caption parameters

    # Training Settings
    args['lr'] = 0.01
    args['weight_decay'] = 0.0001
    args['batch_size'] = 2
    args['use_pretrain_fasterrcnn'] = True
    args['box_detections_per_img'] = 50

    if not os.path.exists(os.path.join(CONFIG_PATH, MODEL_NAME)):
        os.mkdir(os.path.join(CONFIG_PATH, MODEL_NAME))
    with open(os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'), 'w') as f:
        json.dump(args, f, indent=2)

    return args


def save_model(model, optimizer, amp_, results_on_val, iter_counter, flag=None):
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'amp': None,
             'results_on_val': results_on_val,
             'iterations': iter_counter}
    if isinstance(flag, str):
        filename = os.path.join('model_params/' + MODEL_NAME, '{}_{}.pth.tar'.format(MODEL_NAME, flag))
    else:
        filename = os.path.join('model_params/' + MODEL_NAME, '{}.pth.tar'.format(MODEL_NAME))
    print('Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)


# def train(args):
if __name__ == '__main__':
    args = set_args()
    print('Model {} start training...'.format(MODEL_NAME))
    model, transforms_train, transforms_val, collator = get_backbone()

    # if True:
    #     checkpoint = torch.load(
    #         'model_params/train_all_val_all_bz_2_epoch_10_inject_init_attr/train_all_val_all_bz_2_epoch_10_inject_init_attr_best.pth.tar')
    #     model.load_state_dict(checkpoint['model'],strict=False)
    # for name, para in model.named_parameters():
    #     if 'box_describer' not in name:
    #         para.requires_grad = False
    model.to(device)

    optimizer = torch.optim.Adam([{'params': (para for para in model.caption.parameters()
                                              if para.requires_grad)}],
                                 lr=args['lr'], weight_decay=args['weight_decay'])

    lr_warm_up_schedule = [0.1, 0.3, 0.5, 0.7, 0.9]
    lr_d = lr_schedule_cosine(lr_warm_up_schedule, MAX_EPOCHS - len(lr_warm_up_schedule) - 1, T_mult=1, eta_max=1,
                              eta_min=0.1, restart=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_d.get_lr)
    # apex initialization
    opt_level = 'O1'
    # model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    # ref: https://github.com/NVIDIA/apex/issues/441
    # model.roi_heads.box_roi_pool.forward = \
    #     amp.half_function(model.roi_heads.box_roi_pool.forward)

    dictionary = Dictionary.load_from_file(os.path.join(IMG_DIR_ROOT, 'features/dictionary.pkl'))
    train_set = ClevrDenseCapDataset(IMG_DIR_ROOT,dictionary,dataset_type='train',transform=transforms_train)
    val_set = ClevrDenseCapDataset(IMG_DIR_ROOT,dictionary,dataset_type='val',transform=transforms_val)

    idx_to_token = train_set.dictionary.idx2word

    if MAX_TRAIN_IMAGE > 0:
        train_set = Subset(train_set, range(MAX_TRAIN_IMAGE))
    if MAX_VAL_IMAGE > 0:
        val_set = Subset(val_set, range(MAX_VAL_IMAGE))

    train_loader = DataLoaderPFG(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=4,
                                 pin_memory=True, collate_fn=AttrDenseCapDataset.collate_fn)

    iter_counter = 0
    best_map = 0.
    results = None

    # use tensorboard to track the loss
    if USE_TB:
        writer = SummaryWriter()

    # results = quantity_check(model, val_set, idx_to_token, device, max_iter=-1, verbose=True)

    for epoch in range(MAX_EPOCHS):
        with tqdm(total=len(train_loader), ncols=100) as t:
            for batch, (img, targets, info, _) in enumerate(train_loader):
                optimizer.zero_grad()
                img = [img_tensor.to(device) for img_tensor in img]
                # targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
                targets = [tag.to(device) for tag in targets]
                model.train()
                model.attribute.eval()
                losses = model(img, targets)
                loss = losses['loss_objectness'] + losses['loss_rpn_box_reg'] + \
                              losses['loss_classifier'] + losses['loss_box_reg']

                # record loss
                if USE_TB:
                    writer.add_scalar('batch_loss/detect_loss', loss.item(), iter_counter)

                    writer.add_scalar('details/loss_objectness', losses['loss_objectness'].item(), iter_counter)
                    writer.add_scalar('details/loss_rpn_box_reg', losses['loss_rpn_box_reg'].item(), iter_counter)
                    writer.add_scalar('details/loss_classifier', losses['loss_classifier'].item(), iter_counter)
                    writer.add_scalar('details/loss_box_reg', losses['loss_box_reg'].item(), iter_counter)

                if iter_counter % (len(train_set) / (args['batch_size'] * 16)) == 0:
                    print("[{}][{}]\ntotal_loss {:.3f}".format(epoch, batch, loss.item()))
                    for k, v in losses.items():
                        print(" <{}> {:.3f}".format(k, v))

                # total_loss.backward()
                # apex backward
                # with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                loss.backward()
                optimizer.step()
                if iter_counter > 0 and iter_counter % 10000 == 0:
                    try:
                        results = quantity_check(model, val_set, idx_to_token, device, max_iter=-1, verbose=True)
                        if results['map'] > best_map:
                            best_map = results['map']
                            save_model(model, optimizer, None, results, iter_counter)

                        if USE_TB:
                            writer.add_scalar('metric/map', results['map'], iter_counter)
                            writer.add_scalar('metric/det_map', results['detmap'], iter_counter)
                            writer.add_scalar('metric/meteor', results['meteor'], iter_counter)
                        scheduler.step()
                    except AssertionError as e:
                        print('[INFO]: evaluation failed at epoch {}'.format(epoch))
                        print(e)

                iter_counter += 1
                t.set_description('epoch{}'.format(epoch))
                t.set_postfix(total_loss='{:^7.3f}'.format(loss.item()),
                              loss_cap='{:^7.3f}'.format(losses['loss_caption'].item()),
                              lr='{:^7.6f}'.format(optimizer.param_groups[-1]['lr']))
                t.update(1)

        save_model(model, optimizer, None, results, iter_counter, flag=str(epoch))
        # try:
        #     results = quantity_check(model, val_set, idx_to_token, device, max_iter=-1, verbose=True)
        #     if results['map'] > best_map:
        #         best_map = results['map']
        #         save_model(model, optimizer, None, results, iter_counter, flag='best')
        #
        #     if USE_TB:
        #         writer.add_scalar('metric/map', results['map'], iter_counter)
        #         writer.add_scalar('metric/det_map', results['detmap'], iter_counter)
        #         writer.add_scalar('metric/cider', results['cider'], iter_counter)
        # except AssertionError as e:
        #     print('[INFO]: evaluation failed at epoch {}'.format(epoch))
        #     print(e)
        writer.add_scalar('lr/lrd', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('lr/lrc', optimizer.param_groups[-1]['lr'], epoch)
        # scheduler.step()
    save_model(model, optimizer, None, results, iter_counter, flag='end')

    if USE_TB:
        writer.close()

# if __name__ == '__main__':
#     args = set_args()
#     train(args)
