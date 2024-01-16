import os, torch, random, cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataloader import RoadsDataset
from data_visualization import visualize, reverse_one_hot, colour_code_segmentation, crop_image
from augmentation import get_validation_augmentation, get_preprocessing

SHOW_PREDICTIONS = False
# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = './data/tiff/'
x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')

# Load best saved model checkpoint from the current run
if os.path.exists('./best_model.pth'):
    best_model = torch.load('./best_model.pth', map_location = DEVICE)
    print('Loaded UNet model from this run.')

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

test_dataset = RoadsDataset(
    x_test_dir, 
    y_test_dir, 
    augmentation = get_validation_augmentation(), 
    preprocessing = get_preprocessing(preprocessing_fn),
    class_rgb_values = select_class_rgb_values,
)

test_dataloader = DataLoader(test_dataset)

# Test dataset for visualization (without preprocessing transformations)
test_dataset_vis = RoadsDataset(
    x_test_dir, y_test_dir, 
    augmentation = get_validation_augmentation(),
    class_rgb_values = select_class_rgb_values,
)

# Get a random test image / mask index
random_idx = random.randint(0, len(test_dataset_vis) - 1)
image, mask = test_dataset_vis[random_idx]

sample_preds_folder = 'sample_predictions/'
if not os.path.exists(sample_preds_folder):
    os.makedirs(sample_preds_folder)

# Define loss function
loss = smp.utils.losses.DiceLoss()

# Define metrics
metrics = [
    smp.utils.metrics.IoU(threshold = 0.5),
]


for idx in range(len(test_dataset)):

    image, gt_mask = test_dataset[idx]
    image_vis = crop_image(test_dataset_vis[idx][0].astype('uint8'))
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    # Predict test image
    pred_mask = best_model(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    # Convert pred_mask from `CHW` format to `HWC` format
    pred_mask = np.transpose(pred_mask, (1, 2, 0))
    # Get prediction channel corresponding to road
    pred_road_heatmap = pred_mask[:,:,select_classes.index('road')]
    pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values))
    # Convert gt_mask from `CHW` format to `HWC` format
    gt_mask = np.transpose(gt_mask, (1, 2, 0))
    gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values))
    cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1])
    
    if SHOW_PREDICTIONS:
        visualize(
            original_image = image_vis,
            ground_truth_mask = gt_mask,
            predicted_mask = pred_mask,
            predicted_road_heatmap = pred_road_heatmap
        )

### Model Evaluation On Test Dataset ###
test_epoch = smp.utils.train.ValidEpoch(
    model,
    loss = loss, 
    metrics = metrics, 
    device = DEVICE,
    verbose = True,
)

valid_logs = test_epoch.run(test_dataloader)
print("Evaluation on Test Data: ")
print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")

