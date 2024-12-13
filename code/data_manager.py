
# Load images
images = image_list(images_dir)
# load labels
# PDL1
metadata = pd.read_csv(metadata_dir)
PDL1_score = metadata["PDL1_score"]


# length of images in the list
print("Length of list with images:", len(images))
# Length of PDL1_score labels
print("Length of list with images:", len(PDL1_score))
# Shape of the individual images
print("Shape of the 1st image:", images[0].shape) # 46 Channels, 224 (x) * 224 (y) pixels
# Data type 
print("Data type of 1st image:", images[0].dtype)

import os
from imc_preprocessing import IMCPreprocessor
import sys
import gc
import numpy as np
import pandas as pd
from tifffile import tiff

class DataSet:
    """
    Dataset loader and preprocessor
    """
    __base_dir = os.getcwd()
    __image_dir = '../IMC_images' 
    __metadata_dir = '../metadata.csv' 
    __panel_dir = '../panel.csv'

    def __init__(self):
        print("Loading dataset...")

    # Preprocessing (if needed)
    def preprocessing(self, image, transpose=True, normalize=True) -> np.ndarray:
        if transpose:
            return np.transpose(image, (1, 2, 0))
        if normalize:
            return IMCPreprocessor.normalize_multichannel_image(image)
        
    # Load images
    def load_image(self, image_path) -> np.ndarray:
        image = tiff.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image

    # Define a function to create a list of images from files within a folder 
    def image_list(self):
        # List all files in the directory
        image_files = [f for f in os.listdir(self.__image_dir) if os.path.isfile(os.path.join(self.__image_dir, f))]  
        # Initialize a list to store the images
        images = []
        # Loop through each file and read the image
        for image_file in image_files:
            image_path = os.path.join(self.__image_dir, image_file)
            image = self.load_image(image_path)
            images.append(image)        
        return images
    
    # # Extract channel information
    # panel_df = pd.read_csv('panel.csv')
    # channel_names = dict(zip(panel_df['clean_target'].to_list(), panel_df['channel'].to_list()))

    # # change 'your_image_array' to your actual array name that contains the images
    # img_nb = 1
    # channel_to_display = ['Yb173'] # choose a channel to display
    # plt.imshow(IMCPreprocessor.drop_channels(your_image_array[img_nb], channel_to_display, list(channel_names.values()))[0], cmap='gray')
    # plt.title(f"{channel_to_display[0]} - channel")
    # plt.axis('off')
    # plt.show()