import os

# Absolute path to the module directory
package_dir = os.path.dirname(os.path.abspath(__file__))

# GAN
OUTPUT_CHANNELS = 3
LAMBDA = 100

# Image
IMG_WIDTH = 1024
IMG_HEIGHT = 1024

# Train
BRIGHTNESS_DELTA = 0.2 # max brightness delta
EPOCHS = 300
BUFFER_SIZE = 20000 # greater than or equal to the full size of the dataset
BATCH_SIZE = 1
TRAIN_DATASET_PATH = os.path.join(package_dir, '..', 'Dataset', 'train', '*.jpg')
TEST_DATASET_PATH = os.path.join(package_dir, '..', 'Dataset', 'test', '*.jpg')
LOG_DIR = os.path.join(package_dir, 'logs', 'fit')
CHECKPOINT_DIR = os.path.join(package_dir, 'training_checkpoints')
CHECKPOINT_IMG_DIR = os.path.join(package_dir, CHECKPOINT_DIR, 'sample-output')
CONTINUE_FROM_LAST_CHECK = True
CHECKPOINT_EVERY_EPOCH = 1
