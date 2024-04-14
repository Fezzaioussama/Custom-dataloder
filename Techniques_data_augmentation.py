import numpy as np
import random
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Padding image by adding in the border pixels with level 0 (black)
def manual_padding(image, pad_size):
    image = Image.fromarray(image) # convert image from type to 
    width, height = image.size # get the width and height of image 
    padded_image = Image.new('RGB', (width + 2 * pad_size, height + 2 * pad_size), (0, 0, 0)) # create frame of image with size of image plus the pad_size
    padded_image.paste(image, (pad_size, pad_size)) # paste the image to the frame create it 
    return padded_image

pad_size = 10 

example_image = Image.open("D:\Python_code\datasetecocompteur\\20220901_100640_03DD_PortedOrleans-10052_1432_2_clst.png")
image_array = np.array(example_image)
padded_image = manual_padding(image_array, pad_size)

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(example_image)  
plt.subplot(2, 2, 2)
plt.title('Padded Image')
plt.imshow(padded_image)
plt.show()

# Create an example image (replace this line with loading your image)
example_image = Image.open("/content/drive/MyDrive/Test_eco_compteur/datasetecocompteur/20220901_100640_03DD_PortedOrleans-10052_1432_2_clst.png")

# Define the transformations
transform = transforms.Compose([
              transforms.RandomRotation(10),
              transforms.RandomVerticalFlip(p=random.uniform(0, 0.2)),
              transforms.RandomHorizontalFlip(p=random.uniform(0, 0.2)),
              transforms.ColorJitter(brightness=random.uniform(0, 0.5), contrast=random.uniform(0, 0.5),saturation=random.uniform(0, 0.5), hue=random.uniform(0, 0.5)),
              transforms.ToTensor(),
            ])

# Transformed image
transformed_image = transform(example_image)
transformed_image = transformed_image.permute(1, 2, 0) 
transformed_image = transformed_image.numpy()
# real image 
image_array = np.array(example_image)
# display
plt.subplot(2, 2, 3)
plt.title('Original Image')
plt.imshow(example_image)
plt.subplot(2, 2, 4)
plt.title('transformed_image')
plt.imshow(transformed_image)
plt.show()