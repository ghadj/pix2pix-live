import image as im
import gan

import os
import time
import datetime
import tensorflow as tf

EPOCHS = 300
BUFFER_SIZE = 400
BATCH_SIZE = 1
DATASET_PATH = '../dataset/'
LOG_DIR = "logs/"
CHECKPOINT_DIR = './training_checkpoints'
CHECKPOINT_IMG_DIR = CHECKPOINT_DIR + '/img/'


@tf.function
def train_step(input_image, target, epoch, loss_object):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator(
            [input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = gan.generator_loss(
            disc_generated_output, gen_output, target, loss_object)
        disc_loss = gan.discriminator_loss(
            disc_real_output, disc_generated_output, loss_object)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(train_ds, epochs, test_ds):

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    for epoch in range(epochs):
        start = time.time()

        # display.clear_output(wait=True)

        for example_input, example_target in test_ds.take(1):
            gan.generate_images(generator, example_input, example_target,
                                CHECKPOINT_IMG_DIR + '{0:03d}'.format(epoch) + '.png')
        print("Epoch: ", epoch)

        # Train
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            train_step(input_image, target, epoch, loss_object)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time()-start))
    checkpoint.save(file_prefix=checkpoint_prefix)


summary_writer = tf.summary.create_file_writer(
    LOG_DIR + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Input pipeline
train_dataset = tf.data.Dataset.list_files(DATASET_PATH+'train/*.jpg')
train_dataset = train_dataset.map(im.load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(DATASET_PATH+'test/*.jpg')
test_dataset = test_dataset.map(im.load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


# Generator
generator = gan.Generator()
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

# Test generator
# gen_output = generator(inp[tf.newaxis, ...], training=False)
# plt.imshow(gen_output[0, ...])


# Discriminator
discriminator = gan.Discriminator()
# tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

# Test discriminator
# disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
# plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
# plt.colorbar()


# Optimizer
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# Checkpoints
checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))

fit(train_dataset, EPOCHS, test_dataset)
