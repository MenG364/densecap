import torch
from tqdm import tqdm

from utils.data_loader import DenseCapDataset, DataLoaderPFG
from model.evaluator import DenseCapEvaluator


def quality_check(model, dataset, idx_to_token, device, max_iter=-1):
    model.to(device)
    data_loader = DataLoaderPFG(dataset, batch_size=1, shuffle=False, num_workers=1,
                                pin_memory=True, collate_fn=DenseCapDataset.collate_fn)

    print('[quality check]')
    for i, (img, targets, info) in enumerate(data_loader):

        img = [img_tensor.to(device) for img_tensor in img]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        with torch.no_grad():
            model.eval()
            model.return_features = False
            detections = model(img)

        for j in range(len(targets)):
            print('<{}>'.format(info[j]['file_name']))
            print('=== ground truth ===')
            for box, cap, cap_len in zip(targets[j]['boxes'], targets[j]['caps'], targets[j]['caps_len']):
                print('box:', box.tolist())
                print('len:', cap_len.item())
                print('cap:', ' '.join(idx_to_token[idx] for idx in cap.tolist() if idx_to_token[idx] != '<pad>'))
                print('-' * 20)

            print('=== predict ===')
            for box, cap, score in zip(detections[j]['boxes'], detections[j]['caps'], detections[j]['scores']):
                print('box:', [round(c, 2) for c in box.tolist()])
                print('score:', round(score.item(), 2))
                print('cap:', ' '.join(idx_to_token[idx] for idx in cap.tolist() if idx_to_token[idx] != '<pad>'))
                print('-' * 20)

        if i >= max_iter > 0:
            break


def quantity_check(model, dataset, idx_to_token, device, max_iter=-1, verbose=True):
    model.to(device)
    data_loader = DataLoaderPFG(dataset, batch_size=1, shuffle=False, num_workers=0,
                                pin_memory=True, collate_fn=DenseCapDataset.collate_fn)

    evaluator = DenseCapEvaluator(list(model.caption.box_describer.special_idx.keys()))

    print('[quantity check]')

    with tqdm(enumerate(data_loader), total=len(data_loader)) as t:
        for i, (img, targets, info,_) in enumerate(data_loader):

            img = [img_tensor.to(device) for img_tensor in img]
            targets=[tag.to(device) for tag in targets]

            with torch.no_grad():
                model.eval()
                model.return_features = False
                detections = model(img)

            for j in range(len(targets)):
                scores = detections[j].extra_fields['scores']
                boxes = detections[j].bbox
                text = [' '.join(idx_to_token[idx] for idx in cap.tolist() if idx_to_token[idx] != '<pad>')
                        for cap in detections[j].extra_fields['caps']]
                target_boxes = targets[j].bbox
                target_text = [' '.join(idx_to_token[idx] for idx in cap.tolist() if idx_to_token[idx] != '<pad>')
                               for cap in targets[j].extra_fields['caps']]
                img_id = info[j]['file_name']
                if len(boxes) != 0 and len(text) != 0 and len(target_text) != 0:
                    evaluator.add_result(scores, boxes, text, target_boxes, target_text, img_id)
                else:
                    pass
                    print('False')

            if i >= max_iter > 0:
                break

            t.update(1)

    results = evaluator.evaluate(verbose)
    if verbose:
        print('MAP: {:.3f} DET_MAP: {:.3f} Cider: {:.3f}'.format(results['map'] * 100, results['detmap'] * 100,
                                                                 results['cider']))

    return results
