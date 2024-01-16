# https://github.com/HardevKhandhar/road-segmentation-image-processing/blob/main/geospatial_roads.ipynb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataloader import RoadsDataset
from augmentation import get_training_augmentation, get_preprocessing, get_validation_augmentation

DATA_DIR = './data/tiff/'

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

### Get classes and their colors ###
class_dict = pd.read_csv("./data/label_class_dict.csv")
# Get class names
class_names = class_dict['name'].tolist()
# Get class RGB values
class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()

print('All dataset classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)

# Useful to shortlist specific classes in datasets with large number of classes
select_classes = ['background', 'road']

# Get RGB values of required classes
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

print('Selected classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)


ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = select_classes
ACTIVATION = 'sigmoid'

# Create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name = ENCODER, 
    encoder_weights = ENCODER_WEIGHTS, 
    classes = len(CLASSES), 
    activation = ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Get train and validation dataset instances
train_dataset = RoadsDataset(
    x_train_dir, y_train_dir, 
    augmentation = get_training_augmentation(),
    preprocessing = get_preprocessing(preprocessing_fn),
    class_rgb_values = select_class_rgb_values,
)

valid_dataset = RoadsDataset(
    x_valid_dir, y_valid_dir, 
    augmentation = get_validation_augmentation(), 
    preprocessing = get_preprocessing(preprocessing_fn),
    class_rgb_values = select_class_rgb_values,
)

# Get train and validation data loaders
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True, num_workers = 4)
valid_loader = DataLoader(valid_dataset, batch_size = 1, shuffle = False, num_workers = 4)

# Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
TRAINING = True

# Set number of epochs
EPOCHS = 5

# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define loss function
loss = smp.utils.losses.DiceLoss()

# Define metrics
metrics = [
    smp.utils.metrics.IoU(threshold = 0.5),
]

# Define optimizer
optimizer = torch.optim.Adam([ 
    dict(params = model.parameters(), lr = 0.00008),
])

# Define learning rate scheduler 
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0 = 1, T_mult = 2, eta_min = 5e-5,
)
    

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss = loss, 
    metrics = metrics, 
    optimizer = optimizer,
    device = DEVICE,
    verbose = True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss = loss, 
    metrics = metrics, 
    device = DEVICE,
    verbose = True,
)

if TRAINING:

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, EPOCHS):

        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')


train_logs_df = pd.DataFrame(train_logs_list)
valid_logs_df = pd.DataFrame(valid_logs_list)
plt.figure(figsize = (20, 8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw = 3, label = 'Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw = 3, label = 'Valid')
plt.xlabel('Epochs', fontsize = 21)
plt.ylabel('IoU Score', fontsize = 21)
plt.title('IoU Score Plot', fontsize = 21)
plt.legend(loc = 'best', fontsize = 16)
plt.grid()
plt.savefig('iou_score_plot.png')
plt.show()