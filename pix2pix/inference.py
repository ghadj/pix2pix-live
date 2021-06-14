import gan
import image as im

import tensorflow as tf
import os

CHECKPOINT_DIR = './training_checkpoints/'
DATASET_PATH = '../dataset/'
BATCH_SIZE = 1



test_dataset = tf.data.Dataset.list_files(DATASET_PATH+'test/*.jpg')
test_dataset = test_dataset.map(im.load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


# Generator
generator = gan.Generator()

# Discriminator
discriminator = gan.Discriminator()

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

# Run the trained model on a few examples from the test dataset
for inp, tar in test_dataset.take(5):
    gan.generate_images(generator, inp, tar, './test.png')
