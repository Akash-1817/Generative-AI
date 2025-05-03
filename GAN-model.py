import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 127.5 - 1.0  # Rescale the images to the range [-1, 1]
x_train = np.expand_dims(x_train, axis=-1)  # Add a channel dimension (for grayscale)

# Parameters
latent_dim = 100  # Latent space dimension (random noise input)
batch_size = 128
epochs = 10
sample_interval = 1000

# Build the Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(28 * 28 * 1, activation='tanh'),  # Output layer: 28x28 image (grayscale)
        layers.Reshape((28, 28, 1))
    ])
    return model

# Build the Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output layer: 1 (real or fake)
    ])
    return model

# Build the GAN (stacked generator and discriminator)
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    return model

# Compile the Discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Compile the GAN
generator = build_generator()
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Training loop
def train_gan(epochs, batch_size, sample_interval):
    # Labels for real and fake images
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train the Discriminator
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_labels)

        # Print the progress
        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")
            sample_images(epoch)

# Function to save and display generated images
def sample_images(epoch):
    noise = np.random.normal(0, 1, (25, latent_dim))
    gen_images = generator.predict(noise)
    gen_images = 0.5 * gen_images + 0.5  # Rescale to [0, 1] range

    # Plotting the generated images in a 5x5 grid
    fig, axs = plt.subplots(5, 5, figsize=(5, 5))
    count = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(gen_images[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1

    plt.show()  # Display the images directly
    plt.close()

# Start training
train_gan(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)

from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Load the Stable Diffusion model from Hugging Face (Publicly available)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=False)

# Move the model to GPU (if available) for faster inference
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Set the text prompt for realistic image generation
prompt = "A photorealistic portrait of a young woman in a future sci fiction city, high detail, soft lighting"

# Generate the image
image = pipe(prompt).images[0]

# Display the image using matplotlib
plt.imshow(image)
plt.axis("off")  # Turn off axes
plt.show()
