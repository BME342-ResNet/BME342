from typing import Tuple, List
import numpy as np


class IMCPreprocessor:
    """
    This is a generic class that contains different preprocessing methods
    that are generally used for masking tissue on IMC images, 
    but can be used for other purposes as well.
    """
    
    @staticmethod
    def drop_channels(image, channels_of_interest, channel_names) -> Tuple[np.ndarray, List]:
        channel_names_new = [i for i in channel_names if i not in channels_of_interest]
        mask = np.zeros(image.shape[2], dtype=bool) # Error: np.zeros(image.shape[0], dtype=bool) and NOT np.zeros(image.shape[2], dtype=bool) (ONLY ERROR IF YOU DONT TRANSPOSE THE IMAGE, OTHERWISE OKEY)
        
        for i in channels_of_interest:
            mask[channel_names.index(i)] = True
        image = image[:, :, mask] # Error: image[mask, :, :] and not image[:, :, mask] ! (ONLY ERROR IF YOU DONT TRANSPOSE THE IMAGE, OTHERWISE OKEY)
        
        return image, channel_names_new

    @staticmethod
    def collapse_channels(image, pooling_type='mean') -> np.ndarray:
        if pooling_type == 'mean':
            collapsed_image = np.mean(image, axis=-1, keepdims=True)
        elif pooling_type == 'max':
            collapsed_image = np.max(image, axis=-1, keepdims=True)
        elif pooling_type == 'min':
            collapsed_image = np.min(image, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}. Choose from 'mean', 'max', or 'min'.")
        
        return collapsed_image

    @staticmethod
    def normalize_multichannel_image(image, arcsin_transform=False, cofactor=1) -> np.ndarray:
        # Initialize the normalized image array
        normalized_image = np.zeros_like(image, dtype=np.float32)

        # Normalize each channel separately
        for i in range(image.shape[2]):
            channel = image[:, :, i]
            if arcsin_transform:
                channel = np.arcsinh(channel / cofactor)
            min_val = np.min(channel)
            max_val = np.max(channel)
            normalized_image[:, :, i] = (channel - min_val) / (max_val - min_val)

        return normalized_image