import tensorflow as tf
import numpy as np
import keras
import os
# import torch

def initialize_notebook(use_gpu=True):
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
        print("GPU disabled. Running on CPU.")
    else:
        print("GPU enabled. Checking for available GPUs...")
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress non-critical warnings

    # Limit the use of memory if GPUs are available and GPU usage is enabled
    if use_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth to avoid pre-allocating all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                # Set a memory limit, for example, 13 GB out of 15 GB
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=13312)]
                )
                
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

        
        else:
            print("No GPUs found. Running on CPU.\n")
            num_cores = os.cpu_count()
            print(f"Number of available CPU cores: {num_cores}")
            # Set the number of threads for intra- and inter-op parallelism
            tf.config.threading.set_intra_op_parallelism_threads(num_cores)
            tf.config.threading.set_inter_op_parallelism_threads(num_cores)
            
    else:
        print("Running on CPU.\n")


    # CUDA check: we check if `TensorFlow` recognizes the CUDA and available GPUs
    print("\nVerifying TensorFlow and PyTorch CUDA setup...")
    print("TensorFlow version:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda()) 
    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

    # # PyTorch check
    # print("\nPyTorch version:", torch.__version__)
    # print("PyTorch detects CUDA:", torch.cuda.is_available())

    # Keras check
    print("\nKeras version:", keras.__version__)

    print("\nEnd checks and initialization.")



# load data 
from skimage.io import imread 
from skimage.transform import resize

# Define a function to resize images
def resize_images(images, new_size):
    resized_images = [resize(img, new_size, anti_aliasing=True) for img in images]
    return np.array(resized_images, dtype='float32')

# Define a function to create a list of images from files within a folder 
def image_list(image_dir):
    # List all files in the directory
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]  
    # Initialize a list to store the images
    images = []
    # Loop through each file and read the image
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = imread(image_path)
        images.append(image)        
    return images 

def create_train_test_sets(folder_directory, new_image_size):
    # Load and resize training pictures
    print("Load and resize training and test pictures")
    
    # Path data train and test
    train_benign = folder_directory + "train/benign/"
    train_malignant = folder_directory + "train/malignant/"
    
    test_benign = folder_directory + "test/benign/"
    test_malignant = folder_directory + "test/malignant/"
    
    X_benign = resize_images(image_list(train_benign), new_image_size)
    X_malignant = resize_images(image_list(train_malignant), new_image_size)
    
    # # Load and resize testing pictures
    X_benign_test = resize_images(image_list(test_benign), new_image_size)
    X_malignant_test = resize_images(image_list(test_malignant), new_image_size)

    print("load and resize completed")

    # Create labels
    y_benign = np.zeros(X_benign.shape[0])
    y_malignant = np.ones(X_malignant.shape[0])
    
    y_benign_test = np.zeros(X_benign_test.shape[0])
    y_malignant_test = np.ones(X_malignant_test.shape[0])
    
    # Merge data 
    X_train_all = np.concatenate((X_benign, X_malignant), axis = 0)
    y_train_all = np.concatenate((y_benign, y_malignant), axis = 0)
    
    X_test = np.concatenate((X_benign_test, X_malignant_test), axis = 0)
    y_test = np.concatenate((y_benign_test, y_malignant_test), axis = 0)
    
    # Shuffle data
    # start always from the same random sequence 
    tf.random.set_seed(893)
    np.random.seed(23)

    s = np.arange(X_train_all.shape[0])
    np.random.shuffle(s)
    X_train_all = X_train_all[s]
    y_train_all = y_train_all[s]
    
    s = np.arange(X_test.shape[0])
    np.random.shuffle(s)
    X_test = X_test[s]
    y_test = y_test[s]
    
    return X_train_all, y_train_all, X_test, y_test


def main():

    print("Run the module helper.py as a standalone script")

    print("Initialize notebook")
    initialize_notebook()

    # Define the new size
    new_image_size = (94, 94) 
    
    # Set Path Directory Dataset
    path_image_folder = '/home/alessandro/jupyter-env/notebooks/deep-learning-biomedicine-hs24/ImagesSkinCancer/'

    X_train_all, y_train_all, X_test, y_test = create_train_test_sets(path_image_folder, new_image_size)

    print(X_train_all.shape)
    print(X_test.shape)
    
if __name__ == "__main__":
    main()

