import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import time

import vae

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()  # load dataset


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')



def log_normal_pdf(sample, mean, log_var, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-log_var) + log_var + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, log_var = model.encode(x)
    z = model.reparameterize(mean, log_var)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, log_var)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def show_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray_r')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


def shuffle_dataset():
    global train_dataset, test_dataset
    # shuffling the data
    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                     .shuffle(train_size).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(test_size).batch(batch_size))


def train_vae():
    global epoch
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            train_step(model, train_x, opt)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, elbo, end_time - start_time))
        show_images(model, epoch, test_sample)

    plt.imshow(display_image(epoch))
    plt.axis('off')


if __name__ == '__main__':
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_size = 60000
    batch_size = 32
    test_size = 10000

    shuffle_dataset()

    # using Adam optimizer
    opt = tf.keras.optimizers.Adam(1e-4)

    epochs = 10
    latent_dim = 2
    num_examples_to_generate = 16
    random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
    model = vae.VAE(latent_dim)

    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]

    show_images(model, 0, test_sample)

    # train the vae
    train_vae()
