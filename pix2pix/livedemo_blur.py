import cv2
import os
import tensorflow as tf
import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# TF allocate less gpu memory
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import gan
import image as im


# Absolute path to the module directory
package_dir = os.path.dirname(os.path.abspath(__file__))

# Generator
generator = gan.Generator()

# Discriminator
discriminator = gan.Discriminator()

# Optimizer
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Checkpoints
CHECKPOINT_DIR = os.path.join(package_dir,'.', 'training_checkpoints')
checkpoint_prefix = os.path.join(CHECKPOINT_DIR, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))

# Video Capture
cap1 = cv2.VideoCapture(0)

# Capture first frame
input_cap = im.grab_frame(cap1)
zoom_size = 120
input_zoom = im.img_zoom(input_cap, (zoom_size,zoom_size))

# Set output window
window_1 = 'closeup'
cv2.namedWindow(window_1, cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
interframe_wait_ms = 10

while True:

    # Adjust brightness
    input_zoom = np.array(tf.image.adjust_brightness(input_zoom, -0.7))

    # Blur image
    input_blur = im.img_blur(input_zoom, (101,101), 100)

    # Prepare image for GAN
    gan_input = im.load_image_livedemo(input_blur)

    # GAN
    output = np.array(gan.generate_images(generator, gan_input) * 0.5 + 0.5) # normalize all postive

    cv2.imshow(window_1, np.array(cv2.cvtColor(output,cv2.COLOR_RGB2BGR)))    # closeup

    # Grab next frame
    input_cap = im.grab_frame(cap1)
    input_zoom = im.img_zoom(input_cap, (zoom_size,zoom_size))

    if cv2.waitKey(interframe_wait_ms) & 0x7F == ord('q'):
        print("Exit requested.")
        break

cv2.destroyAllWindows()
