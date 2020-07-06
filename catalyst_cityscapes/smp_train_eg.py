#!/usr/bin/env python
# coding: utf-8

# Install required libs
#!pip install -U segmentation-models-pytorch albumentations --user
import _init_paths

# print(f"torch: {torch.__version__}, catalyst: {catalyst.__version__}")
from utils.io_util import *
from datasets._preprocess import *
from datasets.pneumonia import Dataset_SMP
# from utils.mpyetric import Dice
from medpy.metric import dc
import catalyst
import collections
from catalyst.dl import utils
SEED = 42
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import PIL
import albumentations as albu
from albumentations.pytorch import ToTensor, ToTensorV2

albu.LongestMaxSize
albu.HorizontalFlip
a = albu.Rotate()
a = albu.RandomCrop()
a = albu.Normalize()
opti = torch.optim.SGD()


print('GPUs', torch.cuda.device_count())

# ## Loading data
# For this example we will use **CamVid** dataset. It is a set of:
#  - **train** images + segmentation masks
#  - **validation** images + segmentation masks
#  - **test** images + segmentation masks
#  
# All images have 320 pixels height and 480 pixels width.
# For more inforamtion about dataset visit http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/.

# In[4]:
#
# mask_dirs = ['GT_mask_lg1234_score', 'GT_mask_pneumonia_score']
#
# DATA_DIR = Path('/data3/shidejun/pneumonia/ready2train')
# IMAGE_DIR = Path('/data3/shidejun/pneumonia/image_links')
# MASK_DIRS = [DATA_DIR/a for a in mask_dirs]
#
# # get data image paths
# get_fn_func = lambda x: [a for a in os.listdir(DATA_DIR) if 'txt' in a and x in a]
# train_paths = [a for f in get_fn_func('train') for a in load_string_list(DATA_DIR/f)]
# val_paths = [a for f in get_fn_func('val') for a in load_string_list(DATA_DIR/f)]
# test_paths = [a for f in get_fn_func('test') for a in load_string_list(DATA_DIR/f)]
#
# print('train\tval\ttest\n%d\t%d\t%d' %(len(train_paths), len(val_paths), len(test_paths)))
#
#
# def image_id_safe(label_id):
#     image_id = Path(label_id).parent/ (re.sub('_\d', '', str(label_id.split(os.sep)[-1])))
#     return image_id
#
# # In[61]:
# targe_classes = ['backgound', 'GGO_median', 'GGO_white', 'CSD_rough', 'CSD_uniform']



targe_classes = ['backgound', 'ggo', 'csd']
ENCODER = 'efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = targe_classes
ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

# device = torch.device('cuda:{}'.format(args.local_rank))

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES),
    activation=ACTIVATION, 
)

# model = nn.DataParallel(model)
# model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
# model = model.to(device)
# model = nn.parallel.DistributedDataParallel(
# model, device_ids=[args.local_rank], output_device=args.local_rank)
# # model.module.decoder.parameters()
#
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
#
# # prepare data
# train_dataset = Dataset_SMP(
#     IMAGE_DIR, MASK_DIRS, train_paths, classes=targe_classes,
#     augmentation=get_training_augmentation(base_size=1024,
#                                            crop_size=(448, 448)),
#     preprocessing=get_preprocessing_smp(preprocessing_fn)
# )
#
# valid_dataset = Dataset_SMP(
#    IMAGE_DIR, MASK_DIRS, val_paths, classes=targe_classes,
#     augmentation=get_validation_augmentation(base_size=512,
#                                            crop_size=(448, 448)),
#     preprocessing=get_preprocessing_smp(preprocessing_fn),
# )
#
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12)
# valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)

#
# loss = smp.utils.losses.DiceLoss()
# metrics = [
#     smp.utils.metrics.IoU(threshold=0.7),
#     Dice(num_class= len(CLASSES))
# ]
#
# optimizer = torch.optim.Adam([
#     dict(params=model.parameters(), lr=0.0001),
# ])
#
#
# # In[64]:
#
#
# # create epoch runners
# # it is a simple loop of iterating over dataloader`s samples
# train_epoch = smp.utils.train.TrainEpoch(
#     model,
#     loss=loss,
#     metrics=metrics,
#     optimizer=optimizer,
#     device=DEVICE,
#     verbose=True,
# )
#
# valid_epoch = smp.utils.train.ValidEpoch(
#     model,
#     loss=loss,
#     metrics=metrics,
#     device=DEVICE,
#     verbose=True,
# )
#
#
# # train model for 40 epochs
# max_score = 0
#
# for i in range(0, 50):
#     print('\nEpoch: {}'.format(i))
#     train_logs = train_epoch.run(train_loader)
#     valid_logs = valid_epoch.run(valid_loader)
#     # do something (save model, change lr, etc.)
#     if max_score < valid_logs['iou_score']:
#         max_score = valid_logs['iou_score']
#         torch.save(model, './best_model.pth')
#         print('Model saved!')
#
#     if i == 25:
#         optimizer.param_groups[0]['lr'] = 1e-5
#         print('Decrease decoder learning rate to 1e-5!')


# ## Test best saved model

# # load best saved checkpoint
# best_model = torch.load('./best_model.pth')
#
#
# # In[ ]:
#
#
# # create test dataset
# test_dataset = Dataset(
#     x_test_dir,
#     y_test_dir,
#     augmentation=get_validation_augmentation(),
#     preprocessing=get_preprocessing(preprocessing_fn),
#     classes=CLASSES,
# )
#
# test_dataloader = DataLoader(test_dataset)
#
#
# # In[ ]:
#
#
# # evaluate model on test set
# test_epoch = smp.utils.train.ValidEpoch(
#     model=best_model,
#     loss=loss,
#     metrics=metrics,
#     device=DEVICE,
# )
#
# logs = test_epoch.run(test_dataloader)
#
#
# # ## Visualize predictions
#
# # In[ ]:
#
#
# # test dataset without transformations for image visualization
# test_dataset_vis = Dataset(
#     x_test_dir, y_test_dir,
#     classes=CLASSES,
# )
#
#
# # In[ ]:
#
#
# for i in range(5):
#     n = np.random.choice(len(test_dataset))
#
#     image_vis = test_dataset_vis[n][0].astype('uint8')
#     image, gt_mask = test_dataset[n]
#
#     gt_mask = gt_mask.squeeze()
#
#     x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
#     pr_mask = best_model.predict(x_tensor)
#     pr_mask = (pr_mask.squeeze().cpu().numpy().round())
#
#     visualize(
#         image=image_vis,
#         ground_truth_mask=gt_mask,
#         predicted_mask=pr_mask
#     )

