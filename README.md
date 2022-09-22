Some files are really too large to upload, so the project here is incomplete. 

To download the whole project, please refer [this BaiDu Net Disk link](https://pan.baidu.com/s/1T7yqJtkoD6_0AU3m-E0O2w) (fetch code: J93P)

Original paper: https://arxiv.org/abs/1908.06954

# 说明

* 环境需求：
  * python 3.6
  * java 1.8.0
  * PyTorch 1.0
  * cider和coco-caption包，已经作为子模块放在project中


- 模型实现的核心部分在`AoANet`文件夹
- `self-critical.pytorch-master` 用于强化学习
- 一些文件由于过大无法上传所以压缩为`.zip`文件，直接在当前目录下解压即可

* 由于**下载和处理后的数据文件**和**代码运行记录文件（log）**较大（共200+GB），所以在提交的代码中没有附上
* `AoANet/vis`中保留了程序的运行结果，`scores`为模型得分，.json文件为输出的caption

# 数据

要有两份数据：image和caption

需要先拿到caption数据

## 1. caption

这里用到了Karpathy split后的caption数据

下载链接：

https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

下载后解压，把`dataset_coco.json`放到`AoANet/data`下，其他的就可以不用了

/

接着进行预处理，命令：

```bash
$ python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```

`prepro_labels.py` 会给词频<=5的词汇打上一个token，然后给剩下的词建立词汇表

图片信息和词汇表会输出到`AoANet/data/cocotalk.json` 

离散化的caption data 会输出到 `AoANet/data/cocotalk_label.h5`

/

plus：使用SCST进行强化学习来优化参数时，需要再进行一个预处理

```sh
$ python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```

## 2. image

直接下载提取好的特征（bottom-up feature）

下载：https://storage.googleapis.com/bottom-up-attention/trainval.zip

下载后保存在文件夹`data/bu_data`中，然后解压：

```sh
unzip trainval.zip
```

/

下载好后进行处理。输入命令（或者调用`AoANet`下的`run_our_bu.sh`）：

```sh
python scripts/our_make_bu_data.py --output_dir data/cocobu
```

调用`our_make_bu_data.py`，输出结果在 `data/cocobu_fc`, `data/cocobu_att` 和 `data/cocobu_box` .

------

# 训练

​	`AoANet`里的几个文件：

1. `train.sh`：用`train.py`进行训练
2. `train-wo-refining.sh`：训练去掉了encoder中aoa层的模型
3. `train-base.sh`：训练没有aoa的模型，即encoder和decoder的aoa层都去掉
4. `train.py`：训练的全流程
5. `opts.py`：模型参数设置

调用1、2或3即可训练

------

# 测试

注意需要在`coco-caption`目录下下载 Google News negative 300 word2vec model，以及还需要the [Stanford CoreNLP 3.6.0](http://stanfordnlp.github.io/CoreNLP/index.html) code and models

可以直接运行：

```sh
bash get_stanford_models.sh
bash get_google_word2vec_model.sh
```

运行完后应该在`coco-caption\pycocoevalcap\wmd\data`目录下有文件 `GoogleNews-vectors-negative300.bin`，以及在`coco-caption\pycocoevalcap\spice\lib`目录下增加了两个文件`stanford-corenlp-3.6.0.jar` 和 `stanford-corenlp-3.6.0-models.jar`

/

输入命令（或者调用`AoANet`下的`evaluation.sh`）：

```SH
CUDA_VISIBLE_DEVICES=7 python eval.py 
--model log/log_aoanet_rl/model.pth 
--infos_path log/log_aoanet_rl/infos_aoanet.pkl  
--dump_images 0 --dump_json 1 --num_images -1 
--language_eval 1 --beam_size 2 --batch_size 100 
--split test > vis/scores/score_aoa_rl.txt
```

测试结果保存在`AoANet/vis/scores`下

（上述命令为测试完整带AoA的模型。其他模型训练只需改变相应参数，详细可见`evaluation.sh`）

------

# 在新的图片上做captioning

把想要产生caption的图片放在`AoANet/image/new_image`下

输入命令（或者调用`AoANet`下的`generateCaption.sh`）：

```sh
export CUDA_VISIBLE_DEVICES=7 
python generateCaption.py 
--model log/log_base_rl/model.pth --infos_path log/log_base_rl/infos_base.pkl  
--dump_images 0 --dump_json 1 --num_images -1 
--language_eval 0 --beam_size 2 --batch_size 1
python generateCaption.py 
--model log/log_aoanet_rl/model.pth --infos_path log/log_aoanet_rl/infos_aoanet.pkl  
--dump_images 0 --dump_json 1 --num_images -1 
--language_eval 0 --beam_size 2 --batch_size 1
```

上述命令会分别产生base模型和AoANet模型的caption输出

输出结果保存在`AoANet/new_image/new_vis`下
