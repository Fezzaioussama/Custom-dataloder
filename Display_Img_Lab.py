import pandas as pd
from PIL import Image
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
# Load Excel file
file_path = 'D:\Python_code\datasetecocompteur\\annotations.xlsx'
df = pd.read_excel(file_path)
# get path of folder of images 
root_dir = 'D:\Python_code\datasetecocompteur'
l=len(df)

# take idx of image randomly 
rand_indx=random.randint(0,l-1)
# find the name of image and the path of it
img_name = df.iloc[rand_indx, 1]
img_path = f"{root_dir}\{img_name}"
image = Image.open(img_path)
label = torch.tensor(df.iloc[rand_indx, 2:], dtype=torch.long)
# Define class names based on your DataFrame columns
class_names = ['m-loc', 'e-loc', 'meca', 'elec', 'nn_id', 'trot-loc', 'trot']
# display the path of this image 
print('The path of this image is:',img_path)
# display image
image = np.array(image)
plt.imshow(image)
#predict label of this image
predicted_class_name = class_names[torch.argmax(label).item()]
print(predicted_class_name)
plt.title(f"Label: {predicted_class_name}")