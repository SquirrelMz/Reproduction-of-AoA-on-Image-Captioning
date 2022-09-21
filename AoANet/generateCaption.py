# 非原始文档，参考eval.py
# 去除不必要输出
# 改变caption文件保存路径
# 需要生成caption的图片放在new_image/image
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
from dataloadernew import *
import eval_utils
import argparse
import misc.utils as utils
import torch
import sys


def isImage(f):
    supportedExt = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM']
    for ext in supportedExt:
        start_idx = f.rfind(ext)
        if start_idx >= 0 and start_idx + len(ext) == len(f):
            return True
    return False


def create_image_list(image_folder_path):
    n = 1
    image_data = list()
    for root, dirs, files in os.walk(image_folder_path, topdown=False):
        for file in files:
            image_data.append(file)
    return image_data


# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                    help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str, default='resnet101',
                    help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                    help='path to infos to evaluate')
parser.add_argument('--image_folder_root', type=str, default='new_image',
                    help='root of the new image folder')
opts.add_eval_options(parser)

opt = parser.parse_args()
opt.language_eval = 0  # 新图片，不一定适用于coco-caption，强制关闭language-eval
opt.split = 'new'  # 表示这是自定义的数据集（图片需要在coco内）

print("Captioning with model: {}".format(opt.model))
# 为避免更改过多项目主要文件，造成冗余，直接重定向输出，实现输出的过滤
savedStdOut = sys.stdout
outfile = open(os.path.join(opt.image_folder_root, 'others/out'), 'w')
sys.stdout = outfile

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

vocab = infos['vocab']  # ix -> word mapping

# Setup the model
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()

# Create the Data Loader instance
image_list = create_image_list(os.path.join(opt.image_folder_root, "image"))
loader = DataLoaderNew(opt, image_list)
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']
print("Model loaded successfully")

# Set sample options
opt.datset = opt.input_json
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader,
                                                            vars(opt))

# 恢复输出
sys.stdout = savedStdOut
outfile.close()
for i in range(len(split_predictions)):
    print("image {}: {}".format(i, split_predictions[i]['caption']))

# print('loss: ', loss)
# if lang_stats:
#  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('new_image/new_vis/vis.json', 'w'))
