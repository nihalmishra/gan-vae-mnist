import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
from tensorflow.keras import layers
import time

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()  # load dataset

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')


# compute discriminator loss
def calc_disc_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# compute generator loss
def calc_gen_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def define_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # output layer
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    return model

def define_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# take a training step, calculate loss and update weights
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = calc_gen_loss(fake_output)
        disc_loss = calc_disc_loss(real_output, fake_output)
        print('Generator Loss=', gen_loss)
        print('Discriminator Loss=', disc_loss)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        show_images(generator,
                    epoch + 1,
                    seed)

        print('Epoch {} training time is {} sec'.format(epoch + 1, time.time() - start))

    show_images(generator,
                epochs,
                seed)


def show_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray_r')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


if __name__ == '__main__':

    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # using Adam optimizer for both
    gen_opt = tf.keras.optimizers.Adam(1e-4)
    disc_opt = tf.keras.optimizers.Adam(1e-4)

    # Normalize the images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # init loss function
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # initialize generator
    generator = define_generator()
    generator.summary()

    # initialize discriminator
    discriminator = define_discriminator()
    discriminator.summary()

    # train the models
    train(train_dataset, EPOCHS)

    # display the output images
    display_image(EPOCHS)
