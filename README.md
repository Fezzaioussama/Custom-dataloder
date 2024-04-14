# Solution of Test technique "Creation DataLoder" 
In this solution there are three Script each one for specific work

1. PropreDataloder.py: Create Propre Dataloder for training classifier

2. Display_Img_Lab.py: dsipaly image with its propre label 

3. Techniques_data_augmenetation.py: Display Images using technique for data augementation use it for showing theirs works


# PropreDataloder.py

This is a Python code snippet that defines a custom PyTorch dataset class named `PropreDataset` for classification images in classes idicate type of bicycle. The dataset is intended for training a neural network using PyTorch's DataLoader. The dataset is loaded from an Excel file containing image information and labels. Various image transformations and augmentations are applied to the dataset.

## Requirements
- pandas
- numpy
- torch
- torchvision
- Pillow (PIL)

## Usage
1. Install the required dependencies:

    ```bash
    pip install pandas numpy torch torchvision Pillow
    ```

2. Save your dataset annotations in an Excel file, and set the path to the file in the `file_path` variable.

3. Set the `root_dir` variable to the path of the folder containing the dataset images.

4. Adjust the batch size (`batch_size`) according to your system's memory.

5. Run the script.

## Notes

1. Make sure to customize the file paths according to your dataset location.
2. Adjust the transformation parameters based on your specific task and dataset characteristics.

# Display_Img_Lab.py

This Python script randomly selects an image from a dataset described in an Excel file and displays the image along with its label. The script utilizes the `Pandas` library for handling Excel data, the `PIL` (Pillow) library for image manipulation, and `matplotlib` for visualizing the image.

## Requirements
- pandas
- Pillow (PIL)
- torch
- matplotlib
- numpy

## Usage
1. Install the required dependencies:

    ```bash
    pip install pandas Pillow torch matplotlib numpy
    ```

2. Set the `file_path` variable to the path of your dataset annotations Excel file.

3. Set the `root_dir` variable to the path of the folder containing the dataset images.

4. Run the script.

## Notes

1. Customize the file_path and root_dir variables based on your dataset location.

2. This script randomly selects an image and predicts its label. Adjustments can be made to loop through multiple images or perform more complex operations based on your requirements.

# Techniques_data_augmenetation.py

This Python script demonstrates techniques for data augmentation : manual padding and random transformations from library Pytorch. The script uses the `PIL` (Pillow) library for image manipulation, `numpy` for array operations, `random` for randomization, `transforms` from `torchvision` for data augmentation, and `matplotlib.pyplot` for visualization.
in this case i use technique for Data augmentation:
1. Random Rotation
2. Random Horizontal Flip
3. Random Vertical Flip
4. Introduces random color variations to the image.

## Notes

1. Adjust the paths to your images accordingly.
2. Customize the transformation parameters based on your specific task and dataset characteristics.
3. The manual padding function adds black borders to the image, which may be useful in certain scenarios.