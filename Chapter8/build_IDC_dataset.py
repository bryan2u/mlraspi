# Import the required libraries
from config_IDC import config_IDC
from imutils import paths
import random
import shutil
import os
# Grab the paths to all input images in the original input
# directory and shuffle them.
imagePaths = list(paths.list_images(config_IDC.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)
# Compute the training and testing split.
i = int(len(imagePaths) * config_IDC.TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]
# Use part of the training data for validation.
i = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]
# Define the datasets that are built.
datasets = [
    ("training", trainPaths, config.TRAIN_PATH),
    ("validation", valPaths, config.VAL_PATH),
    ("testing", testPaths, config.TEST_PATH)
]
# Loop over the datasets.
for (dType, imagePaths, baseOutput) in datasets:
    # Show which data split created
    print("[INFO] building '{}' split".format(dType))
    # If the base output directory does not exist,
    # create it.
    if not os.path.exists(baseOutput):
        print("[INFO] 'creating {}' directory".format(baseOutput))
        os.makedirs(baseOutput)
    # Loop over the input image paths.
    for inputPath in imagePaths:
        # Extract the filename of the input image and extract
        # the class label ("0" for "negative" and "1" for
        # "positive").
        filename = inputPath.split(os.path.sep)[-1]
        label = filename[-5:-4]
        # Build the path to the label directory.
        labelPath = os.path.sep.join([baseOutput, label])
        # If the label output directory does not exist, create
        # it.
        if not os.path.exists(labelPath):
            print("[INFO] 'creating {}' directory".format(labelPath))
            os.makedirs(labelPath)
        # Construct the path to the destination image and then
        # copy the image itself.
        p = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, p)
