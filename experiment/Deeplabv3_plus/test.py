# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
import numpy as np
import cv2
import os
import pandas as pd

from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
# import torch.optim as optim
from net.sync_batchnorm.replicate import patch_replication_callback

from torch.utils.data import DataLoader


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def test_net():
    dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test')
    dataloader = DataLoader(dataset, batch_size=cfg.TEST_BATCHES, shuffle=False, num_workers=cfg.DATA_WORKERS)

    net = generate_net(cfg)
    print('net initialize')
    if cfg.TEST_CKPT is None:
        raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')

    print('Use %d GPU' % cfg.TEST_GPUS)
    device = torch.device('cuda')
    if cfg.TEST_GPUS > 1:
        net = nn.DataParallel(net)
        patch_replication_callback(net)
    net.to(device)

    print('start loading model %s' % cfg.TEST_CKPT)
    model_dict = torch.load(cfg.TEST_CKPT, map_location=device)
    net.load_state_dict(model_dict)

    net.eval()
    result_list = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            name_batched = sample_batched['name']
            # row_batched = sample_batched['row']
            # col_batched = sample_batched['col']

            [batch, channel, height, width] = sample_batched['image'].size()
            multi_avg = torch.zeros((batch, cfg.MODEL_NUM_CLASSES, height, width), dtype=torch.float32).cuda()
            for rate in cfg.TEST_MULTISCALE:
                inputs_batched = sample_batched['image_%f' % rate]
                predicts = net(inputs_batched.cuda())
                predicts_batched = predicts.clone()
                del predicts
                if cfg.TEST_FLIP:
                    inputs_batched_flip = torch.flip(inputs_batched, [3]).cuda()
                    predicts_flip = torch.flip(net(inputs_batched_flip), [3]).cuda()
                    predicts_batched_flip = predicts_flip.clone()
                    del predicts_flip
                    predicts_batched = (predicts_batched + predicts_batched_flip) / 2.0

                predicts_batched = F.interpolate(predicts_batched, size=None, scale_factor=1/rate, mode='bilinear', align_corners=True)
                multi_avg = multi_avg + predicts_batched
                del predicts_batched

            multi_avg = multi_avg / len(cfg.TEST_MULTISCALE)
            multi_avg = nn.Softmax(dim=1)(multi_avg)
            result = torch.argmax(multi_avg, dim=1).cpu().numpy().astype(np.uint8)

            for i in range(batch):
                # row = row_batched[i]
                # col = col_batched[i]
                p = result[i, :, :]
                # p = cv2.resize(p, dsize=(col, row), interpolation=cv2.INTER_NEAREST)

                result_list.append({'predict': np.uint8(p*255), 'name': name_batched[i]})

    PATH = dataset.save_result(result_list, cfg.MODEL_NAME)  # the path contains all segmentation predictions
    infile = os.listdir(PATH)
    ID = []
    RLE = []
    for name in infile:
        seg = cv2.imread(PATH + '/' + name)
        seg = cv2.resize(seg, (256, 256), 0, 0, cv2.INTER_NEAREST)[:, :, 0]

        # convert the segmentation predictions into binary results
        seg[seg > 125] = 255
        seg[seg <= 125] = 0
        seg[seg > 125] = 1  # np.unique(seg) = [0, 1]; seg.shape = (256, 256)

        ID.append(name.split('.jpg')[0])
        RLE.append(mask2rle(seg))

    dataframe = pd.DataFrame({'ImageId': ID, 'EncodedPixels': RLE})
    dataframe.to_csv("SegTest.csv", index=False, sep=',')  # submit this csv file
    print("SegTest.csv has been saved!")
    print('Test finished!')


if __name__ == '__main__':
    test_net()
