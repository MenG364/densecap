import math

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.image as mpimg


def plot_boxes(img, boxes, savename=None, scores=None, class_names=None):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    # width = img.width
    # height = img.height
    # width = 82
    # height = 139
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        # x1 = (box[0] - box[2] / 2.0) * width
        # y1 = (box[1] - box[3] / 2.0) * height
        # x2 = (box[0] + box[2] / 2.0) * width
        # y2 = (box[1] + box[3] / 2.0) * height

        # x1 = box[0]
        # y1 = box[1] if box[1] > 0 else 0
        # # y1=box[1]
        # # y2=box[3]
        # x2 = (box[0] + box[2])
        # y2 = (box[1] + box[3])

        x1 = box[0]
        x2 = box[2]
        y1 = box[1]
        y2 = box[3]

        # if y2 < 0: y2 = box[3] - 1
        rgb = (255, 0, 0)
        font = ImageFont.truetype(
            font='C:\Windows\Fonts\JetBrainsMono-Regular.ttf',
            size=40
        )
        if scores is not None and class_names is None:
            score = round(scores[i], 2)
            red = get_color(0, score, 1)
            green = get_color(1, score, 1)
            blue = get_color(2, score, 1)
            rgb = (red, green, blue)
            rgb = tuple(np.random.randint(0, 255, size=[3]))
            # 5.获取label长宽
            label_size = draw.textsize("1", font)
            # 6.设置label起点
            text_origin = np.array([x1, y1 + 0.2 * label_size[1]])
            draw.rectangle([x1, y1, x2, y2], outline=rgb, width=4)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=rgb)
            draw.text((x1, y1), str(i), fill=(0, 0, 0), font=font)

        elif len(box) >= 6 and class_names:
            cls_conf = box[4]
            cls_id = box[5]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            label_size = draw.textsize(class_names[cls_id], font)
            # 6.设置label起点
            text_origin = np.array([x1, y1 + 0.2 * label_size[1]])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=rgb)
            draw.rectangle([x1, y1, x2, y2], outline=rgb, width=4)
            draw.text((x1, y1), class_names[cls_id] + ' ' + str(round(cls_conf, 2)), fill=(0, 0, 0), font=font)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename, quality=95, subsampling=1)
    return img


if __name__ == '__main__':
    # img = 'F:\datasets\VQA\VQA2.0old\data\\visual-genome\VG_100K_2/2413345.jpg'
    # img = 'F:\datasets\VQA\VQA2.0old\data\\visual-genome\VG_100K_2/2405347.jpg'
    img = 'F:\\tensorflow_datasets\\downloads\\extracted\\val\\val2014\\COCO_val2014_000000000520.jpg'
    # with open("./output/result1.json",'r') as f:
    #     features = np.array(f.get('image_features')[:125530])
    lena = Image.open(img).convert('RGB')
    box = np.asarray([[
        408.06,
        247.39,
        441.89,
        337.49
    ]], dtype=np.int32)
    scores = [0.5]
    plot_boxes(lena, box, 'pre5.jpg', scores=scores)
