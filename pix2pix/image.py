import cv2
import tensorflow as tf

import params as pm


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, pm.IMG_HEIGHT, pm.IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    # Normalizing the images to [-1, 1]
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # Resizing
    input_image, real_image = resize(input_image, real_image, 1080, 1080)

    # Random cropping
    input_image, real_image = random_crop(input_image, real_image)

    # Random brightness
    delta = tf.random.uniform(shape=[], minval=-pm.BRIGHTNESS_DELTA, maxval=pm.BRIGHTNESS_DELTA)
    input_image = tf.image.adjust_brightness(input_image, delta)
    real_image = tf.image.adjust_brightness(real_image, delta)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     pm.IMG_HEIGHT, pm.IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_livedemo(image):
    input_image = tf.cast(image, tf.float32)
    input_image, _ = resize(input_image, input_image, pm.IMG_HEIGHT, pm.IMG_WIDTH)
    input_image, _ = normalize(input_image, input_image)
    input_image = tf.expand_dims(input_image, 0)
    
    return input_image


def grab_frame(cap):
    ret, frame = cap.read()

    # Crop frame to rectangular shape
    h, w = frame.shape[:2]
    center = (h/2, w/2)
    wh = min(w, h)
    y = center[0] - wh/2
    x = center[1] - wh/2

    frame_crop = frame[int(y):int(y+wh), int(x):int(x+wh)]
    frame_resize = cv2.resize(frame_crop, (pm.IMG_WIDTH,  pm.IMG_HEIGHT))

    return frame_resize


def img_blur(image, kernel_size, sigma):
    # Convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur Image
    img_blur = cv2.GaussianBlur(img_gray, kernel_size, sigma)
    # Convert back to BGR
    img_blur_bgr = cv2.cvtColor(img_blur, cv2.COLOR_GRAY2BGR)

    return img_blur_bgr


def img_zoom(image, size):
    center = tuple(s/2 for s in image.shape[:2])
    w, h = size
    x = center[1] - w/2
    y = center[0] - h/2
    image_zoom = cv2.resize(image[int(y):int(y+h), int(x):int(x+w)], 
                            (pm.IMG_WIDTH,  pm.IMG_HEIGHT))

    return image_zoom