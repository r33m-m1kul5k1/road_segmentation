import torch, cv2, os
from data_visualization import one_hot_encode

class RoadsDataset(torch.utils.data.Dataset):
    """
    Massachusetts Roads Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Arguments:
        images_dir (str) : path to images folder
        masks_dir (str) : path to segmentation masks folder
        class_rgb_values (list) : RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose) : data transfromation pipeline (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose) : data preprocessing (e.g. noralization, shape manipulation, etc.)
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            class_rgb_values = None, 
            augmentation = None, 
            preprocessing = None,
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # Read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        
        # One-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image = image, mask = mask)
            image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image = image, mask = mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        # Return length 
        return len(self.image_paths)