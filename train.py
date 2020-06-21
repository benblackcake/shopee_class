
import tensorflow as tf
import numpy as np
import cv2
from utils import DataSet,read_test_set,read_train_sets



def main():
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',\
      '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',\
      '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', \
      '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', \
      '40', '41']
    num_classes = len(classes)
    '''Configuration and Hyperparameters'''
    # Convolutional Layer 1.
    filter_size1 = 3 
    num_filters1 = 32

    # Convolutional Layer 2.
    filter_size2 = 3
    num_filters2 = 32

    # Convolutional Layer 3.
    filter_size3 = 3
    num_filters3 = 64
    # Fully-connected layer.
    fc_size = 128  # Number of neurons in fully-connected layer.
    # Number of color channels for the images (RGB)
    num_channels = 3

    # image dimensions, square image -> img_size x img_size
    img_size = 28

    # Size of image when flattened to a single dimension
    img_size_flat = img_size * img_size * num_channels

    # Tuple with height and width of images used to reshape arrays.
    img_shape = (img_size, img_size)

    batch_size = 32
    validation_size = .16

    train_path = 'done_dataset/train/'
    test_path = 'done_dataset/test/'
    checkpoint_dir = "models/"

    data = read_train_sets(train_path, img_size, classes, validation_size=validation_size)
    test_images, test_ids = read_test_set(test_path, img_size)






if __name__ == "__main__":
    main()