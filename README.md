## Reproduce Benchmark by Catalyst.DL: PSPNet on Cityscapes

[Catalyst](https://github.com/catalyst-team/catalyst) is an easy-to-use framework for training deep learning models. However, it is essential to prove that it can reproduce the state-of-the-art performance. This reporitory tries to take such step in the field of semantic segmentation. This repo is forked from the official repo of [PSPNet](https://github.com/hszhao/semseg) and supplemented with a folder [catalyst_cityscapes](./catalyst_cityscapes) enabling training PSPNet on the semantic segmentation benchmark Cityscapes using catalyst framework. Right now, this repo with catalyst was only able to achieve an mIoU of 0.739 on the validation dataset of cityscapes, shy a lot comparing to the STOA 0.773 by the official PSPNet. The following section will illustrate what has been tried to mimics the components of official PSPNet in the catalyst training pipeline. Any comments and suggestions are welcomed for improving the performance.

### Requirements

Install additional packages: `albumentations` and `smp`

Clone the repository and install catalyst


```bash
git clone https://github.com/baibaidj/semseg-catalyst.git

pip install -U catalyst[cv] # for bash

pip install -U "catalyst[cv]" # for zsh
```

### get pretrain model
pretrained model should be downloaded and put under [./initmodel/resnet50_v2.pth](https://pan.baidu.com/s/1BdAR5m_14nDksyOkgNYW0w) access code: tcqn

### Get dataset
We only use the fine annotation part of cityscapes dataset, including the fine training and fine validation. The dataset, leftImg8bit_trainvaltest.zip for image and gtFine_trainvaltest.zip (241MB) for gtmask, can be downloaded [here](https://www.cityscapes-dataset.com/downloads/). 


The data folder should look like the following structures:

```
cityscapes/
├── fine_train.txt
├── fine_val.txt
├── gtFine
│   ├── test
│   ├── train
│   └── val
├── leftImg8bit
│   ├── test
│   ├── train
│   └── val
```

A glimpse of the fine_val.txt/fine_train.txt:

```
leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelTrainIds.png
leftImg8bit/val/frankfurt/frankfurt_000000_000576_leftImg8bit.png gtFine/val/frankfurt/frankfurt_000000_000576_gtFine_labelTrainIds.png
leftImg8bit/val/frankfurt/frankfurt_000000_001016_leftImg8bit.png gtFine/val/frankfurt/frankfurt_000000_001016_gtFine_labelTrainIds.png
```

You may put the dataset in place (disk path) different than mine. So you may need to modify the [config file](./catalyst_cityscapes/config-cityscapes-ofPSPNet-resnet50.yml) accordingly to let the training code know where to load the data. 
Specifically, you need to change three parameters, namely data_root, train_set, valid_set:
```
stages:
  data_params:
    data_root: '/mnt/data/cityscapes'  # the root storing the dataset
    train_set: '/mnt/data/cityscapes/fine_train.txt' # abs path to the training list
    valid_set: '/mnt/data/cityscapes/fine_val.txt' # abs path to the validation list
```

### PSPNet model
adopted directly from semseg repo with only one major modification:
remove the criterion method from the model object and make it ouput final prediction and auxillary prediction as tuple. 

The catalyst will use the same criterion (nn.CrossEntropyLoss) outside of the model object to calculate loss and do backprop. 

This is a suspicious place that may cause the difference in performance and need further investigation. 

### Data Augmentation (DA)
TODO: How to utilize the albumentation package to mimics the official DA. 

### Main metric 
TODO: How to calculate mIoU during training to select the best model against validation datset.

### SyncBN
official: model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
catalyst: model = apex.parallel.convert_syncbn_model(model)

### Learning rate schedualer
official: [poly_learning_rate](https://github.com/hszhao/semseg/blob/4f274c3f276778228bc14a4565822d46359f0cc8/util/util.py#L34)

catalyst: [OneCycleLRWithWarmup](https://catalyst-team.github.io/catalyst/api/contrib.html?highlight=onecyclelrwithwarmup#catalyst.contrib.nn.schedulers.onecycle.OneCycleLRWithWarmup), whose behavior is controled by [config file](./catalyst_cityscapes/config-cityscapes-ofPSPNet-resnet50.yml)


### Local run for training

```bash
bash tool\train_catalyst_cityscapes.sh

```

### local run for testing
we use the code of semseg repo for testing, which invovles multi-patch inference for a single image and renders the results strictly comparable to the official ones. 

```bash
bash tool\test_cat.sh
```
