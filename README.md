

## Usage


### 1. Download the datasets

Download the following datasets and unzip them into `data` folder.


* [DUTS](https://drive.google.com/open?id=1immMDAPC9Eb2KCtGi6AdfvXvQJnSkHHo) dataset. The .lst file for training is `data/DUTS/DUTS-TR/train_pair.lst`.

* [Datasets for testing](https://drive.google.com/open?id=1eB-59cMrYnhmMrz7hLWQ7mIssRaD-f4o).

### 2. Download the pre-trained models for backbone

Download the following pre-trained models [GoogleDrive](https://drive.google.com/open?id=1Q2Fg2KZV8AzNdWNjNgcavffKJBChdBgy) | [BaiduYun](https://pan.baidu.com/s/1ehZheaqeU3pyvYQfRU9c6A) (pwd: **27p5**) into `dataset/pretrained` folder. 

### 3. Train

1. Set the `--train_root` and `--train_list` path in `train.sh` correctly.

2. We demo using ResNet-50 as network backbone and train with a initial lr of 5e-5 for 24 epoches, which is divided by 10 after 15 epochs.
```shell
./train.sh
```
3. After training the result model will be stored under `results/run-*` folder.

### 4. Test


```shell
python main.py --mode='test' --model='results/run-*/models/final.pth' --test_fold='results/run-*-sal-e' --sal_mode='e'
```


All results saliency maps will be stored under `results/run-*` folders in .png formats.


### 5. Evaluation results

https://github.com/NathanUA/Binary-Segmentation-Evaluation-Tool. 
The Code was used for evaluation in CVPR 2019 paper 'BASNet: Boundary-Aware Salient Object Detection code', Xuebin Qin, Zichen Zhang, Chenyang Huang, Chao Gao, Masood Dehghan and Martin Jagersand. 
