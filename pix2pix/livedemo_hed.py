import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')

import gan
from hed import edge_detector as ed
import image as im
import params as pm


def quantize_img(image, n=3):
    indices = np.arange(0,256)   # List of all colors
    divider = np.linspace(0,255,n+1)[1] # we get a divider
    quantiz = np.int0(np.linspace(0,255,n)) # we get quantization colors
    color_levels = np.clip(np.int0(indices/divider),0,n-1) # color levels 0,1,2..
    palette = quantiz[color_levels] # Creating the palette
    im2 = palette[image]  # Applying palette on image
    im2 = cv2.convertScaleAbs(im2) # Converting image back to uint8

    return cv2.cvtColor(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def grab_frame(cap):
    ret,frame = cap.read()
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

# Setup GAN model
# Absolute path to the module directory
package_dir = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(package_dir, '..', 'Checkpoints', '07-06')

# Generator
generator = gan.Generator()

# Discriminator
discriminator = gan.Discriminator()

# Optimizer
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Checkpoints
checkpoint_prefix = os.path.join(CHECKPOINT_DIR, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))

# Setup Edge Detector
detector = ed.EdgeDetector()

# Video Capture
cap1 = cv2.VideoCapture(0)

window_name = 'live demo'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
interframe_wait_ms = 30

input = grab_frame(cap1)
previous_img = cv2.resize(input, (pm.IMG_WIDTH,  pm.IMG_HEIGHT))

while True:
    # Image quantization
    input_quant = quantize_img(cv2.resize(input, (pm.IMG_WIDTH,  pm.IMG_HEIGHT)))

    # Use HED model for edge detection
    hed_output = detector.run(input_quant)

    # Use Canny edge detector
    #input_filter = input_quant#cv2.filter2D(input_quant,-1,np.ones((6,6),np.float32)/36)
    #hed_output = cv2.bitwise_not(cv2.cvtColor(cv2.Canny(input_filter, 0, 100), cv2.COLOR_GRAY2BGR))
    #hed_output = cv2.erode(hed_output,  np.ones((3, 3), 'uint8'), iterations=1)
    #hed_output = cv2.filter2D(hed_output,-1,np.ones((6,6),np.float32)/36)

    # Prepare input for GAN
    hed_output = tf.cast(hed_output, tf.float32)
    hed_output, _ = im.resize(hed_output, input, pm.IMG_HEIGHT, pm.IMG_WIDTH)
    hed_output, _ = im.normalize(hed_output, input)
    hed_output = tf.expand_dims(hed_output,0)

    # GAN
    gan_output = np.array(gan.generate_images(generator, hed_output) * 0.5 + 0.5)

    final = gan_output

    cv2.imshow(window_name,cv2.cvtColor(gan_output,cv2.COLOR_RGB2BGR))

    # Grab next frame
    input = cv2.resize(grab_frame(cap1), (pm.IMG_WIDTH,  pm.IMG_HEIGHT))


    if cv2.waitKey(interframe_wait_ms) & 0x7F == ord('q'):
        print("Exit requested.")
        break
