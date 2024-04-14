import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# Create PropreDataset class
class PropreDataset(Dataset):
    def __init__(self, dataframe, root_dir, ttransform=None, max_padding=20):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.ttransform = ttransform
        self.max_padding = max_padding
    
    def random_padding(self, image,max_padding):
        pad_size=random.randint(0,max_padding)
        image = np.array(image)
        image = Image.fromarray(image)
        width, height = image.size
        padded_image = Image.new('RGB', (width + 2 * pad_size, height + 2 * pad_size), (0, 0, 0))
        padded_image.paste(image, (pad_size, pad_size))
        
        return padded_image

    # Know the lenght of dataset
    def __len__(self):
        return len(self.dataframe)
     
    # get item of index=idx in File excel 
    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 1]
        img_path = f"{self.root_dir}/{img_name}"
        image = Image.open(img_path)
        
    # Take random varaible and if it <0.5 applay this transfom technique else don't apply it 
        # Apply random Rotation
        if random.uniform(0, 1) < 0.5:
            image = transforms.RandomRotation(30)(image)
        # Apply random Horizontal Flip
        if random.uniform(0, 1) < 0.5:
            image = transforms.RandomVerticalFlip(p=random.uniform(0, 0.3))(image)
        # Apply random Vertical Flip
        if random.uniform(0, 1) < 0.5:
            image = transforms.RandomHorizontalFlip(p=random.uniform(0, 0.3))(image)
        # Apply random ColorJitter
        if random.uniform(0, 1) < 0.5:
            image = transforms.ColorJitter(brightness=random.uniform(0, 0.5), 
                                           contrast=random.uniform(0, 0.5),
                                           saturation=random.uniform(0, 0.5),
                                           hue=random.uniform(0, 0.5))(image)
        # Apply random padding
        if random.uniform(0, 1) < 0.5:
            image=self.random_padding(image,self.max_padding)

        # Labeling images
        label = torch.tensor(self.dataframe.iloc[idx, 2:], dtype=torch.long)
        # these labeling is depend to index of each class in file Excel:class_names = ['m-loc', 'e-loc', 'meca', 'elec', 'nn_id', 'trot-loc', 'trot']
        labels = [0, 1, 2, 3, 4, 5, 6]
        label = labels[torch.argmax(label).item()]
        label = torch.tensor(label)

        # Resize and nomralize and transfrom image to tensor
        if self.ttransform:
            image = self.ttransform(image)

        return image, label
    
# Define Transfrom for Resize images with same size for use it in batch size and transfrom it to tensors
# Also normalize images for use it in training for get best resualt with ResNet
data_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

## uploading dataset
# Upload File Excel
file_path = 'D:\Python_code\datasetecocompteur\\annotations.xlsx'
df = pd.read_excel(file_path)
# get path of folder of images 
root_dir = 'D:\Python_code\datasetecocompteur'

# Create object dataset from class PropreDataset
dataset = PropreDataset(df, root_dir, ttransform=data_transform)
# Create the DataLoader
batch_size = 10
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)