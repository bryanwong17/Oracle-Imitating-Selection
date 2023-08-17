import torch
import torchvision.transforms as T

from PIL import Image

class PatchesDataset(torch.utils.data.Dataset):
    """
    Helper Class to create the PyTorch dataset
    """

    def __init__(self, df, img_size, is_train=True):
        super().__init__()
        self.df_data = df.values
        self.img_size = img_size
        self.is_train = is_train
        self.transforms = self.get_transforms()
        self.class_mapping = {
            'D': [1, 0, 0],
            'M': [0, 1, 0],
            'N': [0, 0, 1]
        }
    
    def __len__(self):
        return len(self.df_data)
    
    def __labels__(self):
        return self.df_data["oracle_selection"]

    def get_transforms(self):

        # TODO: Applying more random and strong data augmentation to the examples that are limited, and less random augmentation on examples that are quiet enough
        if self.is_train:
            return T.Compose([
                T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomApply([
                    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ], p=0.5),
                T.RandomApply([T.RandomRotation(180)], p=0.5),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            return T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def __getitem__(self, index):
        # image_path from 'image_path' column in dataset_information(modified).csv
        # wsi_gt from 'WSI_GT' column in dataset_information(modified).csv
        # patch_pred from 'prediction' column in dataset_information(modified).csv
        # patch_conf_score from 'Confidence_score' column in dataset_information(modified).csv
        # label from 'oracle_selection' column in dataset_information(modified).csv

        image_path, wsi_gt, patch_pred, patch_conf_score, label = self.df_data[index][-1], self.df_data[index][2], self.df_data[index][5], self.df_data[index][6], self.df_data[index][-2]
        image = Image.open(image_path).convert("RGB")
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        # one hot encode the wsi_gt and patch_pred variables

        wsi_gt_one_hot = torch.tensor(self.class_mapping[wsi_gt])
        patch_pred_one_hot = torch.tensor(self.class_mapping[patch_pred])

        patch_conf_score = torch.tensor(patch_conf_score)

        # 3 + 3 + 1 = 7 dims
        meta_info = [wsi_gt_one_hot, patch_pred_one_hot, patch_conf_score]

        return image, meta_info, label