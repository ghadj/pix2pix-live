import os

# Absolute path to the module directory
package_dir = os.path.dirname(os.path.abspath(__file__))

# GAN
OUTPUT_CHANNELS = 3
LAMBDA = 100

# Image
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Train
EPOCHS = 300
BUFFER_SIZE = 400
BATCH_SIZE = 1
DATASET_PATH = os.path.join(package_dir, '..', 'Dataset')
LOG_DIR = os.path.join(package_dir, 'logs', 'fit')
CHECKPOINT_DIR = os.path.join(package_dir, 'training_checkpoints')
CHECKPOINT_IMG_DIR = os.path.join(package_dir, CHECKPOINT_DIR, 'sample-output')
CONTINUE_FROM_LAST_CHECK = True
