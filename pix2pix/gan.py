from matplotlib import pyplot as plt
import tensorflow as tf

import params as pm


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[pm.IMG_WIDTH, pm.IMG_HEIGHT, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),   # (bs, 512, 512, 64)
        downsample(128, 4),                         # (bs, 256, 256, 128)
        downsample(256, 4),                         # (bs, 128, 128, 256)
        downsample(512, 4),                         # (bs, 64, 64, 521)
        downsample(512, 4),                         # (bs, 32, 32, 512)
        downsample(512, 4),                         # (bs, 16, 16, 512)
        downsample(512, 4),                         # (bs, 8, 8, 512)
        downsample(512, 4),                         # (bs, 4, 4, 512)
        downsample(512, 4),                         # (bs, 2, 2, 512)
        downsample(512, 4),                         # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),       # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),       # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),       # (bs, 8, 8, 1024)
        upsample(512, 4, apply_dropout=True),       # (bs, 16, 16, 1024)
        upsample(512, 4),                           # (bs, 32, 32, 1024)
        upsample(512, 4),                           # (bs, 64, 64, 512)
        upsample(256, 4),                           # (bs, 128, 128, 512)
        upsample(128, 4),                           # (bs, 256, 256, 256)
        upsample(64, 4),                            # (bs, 512, 512, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(pm.OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 1024, 1024, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def generator_loss(disc_generated_output, gen_output, target, loss_object):
    gan_loss = loss_object(tf.ones_like(
        disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (pm.LAMBDA * l1_loss)
    
    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[pm.IMG_WIDTH, pm.IMG_HEIGHT, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[pm.IMG_WIDTH, pm.IMG_HEIGHT, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 1024, 1024, channels*2)

    down1 = downsample(64, 4, False)(x)     # (bs, 512, 512, 64)
    down2 = downsample(128, 4)(down1)       # (bs, 256, 256, 128)
    down3 = downsample(256, 4)(down2)       # (bs, 128, 128, 256)
    down4 = downsample(256, 4)(down3)       # (bs, 64, 64, 256)
    down5 = downsample(256, 4)(down4)       # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down5)          # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)    # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)     # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output, loss_object):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(
        disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generate_images(model, test_input, tar=None, filename=None):
    prediction = model(test_input, training=True)

    if not (tar is None and filename is None):
        fig = plt.figure(figsize=(50, 50))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Initial Image', 'Output Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i], fontsize=20)
            # Set the pixel values between [0, 1] to plot it
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')

        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    return prediction[0]