因为数据规模太大（超过200G），所以不作为附件提交，详情可见`projectREADME`

代码运行后完整的data文件夹内容如图：

<img src="C:\Users\zhang\AppData\Roaming\Typora\typora-user-images\image-20211220154725027.png" alt="image-20211220154725027" style="zoom:50%;" />

其中：

* `bu_data`：下载的image数据，bottom up feature
* `cocobu_xxx` 三个文件夹：`our_make_bu_data.py`的输出结果，即对bottom up feature的预处理
* `coco-train-xxx.p` 两个文件：`prepro_ngrams.py`的输出结果，用于SCST
* `cocotalk.json` & `cocotalk_label.h5`：`prepro_labels.py`的输出结果，即caption数据的预处理
* `dataset_coco.json`：下载的caption数据