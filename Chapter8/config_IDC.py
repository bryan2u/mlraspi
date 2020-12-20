# Import the required library
import os
# Initialize the path to the input image directory.
ORIG_INPUT_DATASET = "dataset"
# Initialize the base path to the directory that contain the
# images remaining after the training and testing splits.
BASE_PATH = "datasets/idc"
# Define the training, validation, and testing directory paths.
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
# Define the data split that will be used for training.
TRAIN_SPLIT = 0.8
# Define the data split that will be used for validation.
VAL_SPLIT = 0.1
# Display to user that configuration is complete.
print('[INFO]: Configuration complete')
