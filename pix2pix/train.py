import os
import time
import datetime
import tensorflow as tf
import tqdm

import gan
import image as im
import params as pm


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

    for epoch in (range(epochs)):
        start = time.time()

        # Test
        for example_input, example_target in test_ds.take(1):
            gan.generate_images(generator, example_input, example_target,
                                os.path.join(pm.CHECKPOINT_IMG_DIR, '{0:03d}'.format(epoch) + '.png'))

        # Train
        for n, (input_image, target) in tqdm.tqdm(train_ds.enumerate(), desc='Epoch {}'.format(epoch)):
            train_step(input_image, target, epoch, loss_object)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

        # Save Checkpoint
        if (epoch + 1) % pm.CHECKPOINT_EVERY_EPOCH == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


summary_writer = tf.summary.create_file_writer(
    pm.LOG_DIR + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

# Input pipeline
train_dataset = tf.data.Dataset.list_files(pm.TRAIN_DATASET_PATH)
train_dataset = train_dataset.shuffle(pm.BUFFER_SIZE)
train_dataset = train_dataset.map(im.load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(pm.BATCH_SIZE).prefetch(1)

test_dataset = tf.data.Dataset.list_files(pm.TEST_DATASET_PATH)
test_dataset = test_dataset.map(im.load_image_test)
test_dataset = test_dataset.batch(pm.BATCH_SIZE)

# Generator
generator = gan.Generator()
tf.keras.utils.plot_model(generator, to_file='generator.png', show_shapes=True, dpi=64)

# Discriminator
discriminator = gan.Discriminator()
tf.keras.utils.plot_model(discriminator, to_file='discriminator.png', show_shapes=True, dpi=64)

# Optimizer
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Checkpoints
checkpoint_prefix = os.path.join(pm.CHECKPOINT_DIR, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Restoring the latest checkpoint in checkpoint_dir
if pm.CONTINUE_FROM_LAST_CHECK:
    checkpoint.restore(tf.train.latest_checkpoint(pm.CHECKPOINT_DIR))

fit(train_dataset, pm.EPOCHS, test_dataset)