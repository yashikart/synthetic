# synthetic

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ----- 1. Load and prepare MNIST data -----
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.reshape(-1, 28*28) - 127.5) / 127.5   # scale to [-1,1]
latent_dim = 100

# ----- 2. Build generator -----
generator = Sequential([
    Dense(128, activation='relu', input_dim=latent_dim),
    Dense(784, activation='tanh')   # output fake 28x28 image
])

# ----- 3. Build discriminator -----
discriminator = Sequential([
    Dense(128, activation='relu', input_dim=784),
    Dense(1, activation='sigmoid')
])
discriminator.compile(optimizer=Adam(0.001), loss='binary_crossentropy')

# ----- 4. Combine into GAN -----
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(optimizer=Adam(0.001), loss='binary_crossentropy')

# ----- 5. Train quickly -----
for epoch in range(1001):
    # real images
    idx = np.random.randint(0, X_train.shape[0], 64)
    real = X_train[idx]
    # fake images
    noise = np.random.normal(0, 1, (64, latent_dim))
    fake = generator.predict(noise)
    # train discriminator
    discriminator.train_on_batch(real, np.ones((64, 1)))
    discriminator.train_on_batch(fake, np.zeros((64, 1)))
    # train generator
    gan.train_on_batch(noise, np.ones((64, 1)))

    if epoch % 200 == 0:
        print(f"Epoch {epoch}")
        # generate and show sample image
        noise = np.random.normal(0, 1, (1, latent_dim))
        gen = generator.predict(noise)
        plt.imshow(gen.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.show()



A Generative Adversarial Network has two parts:

Generator (G): creates fake images from random noise.

Discriminator (D): tries to detect whether an image is real or fake.

They compete like a game:

The Generator improves to fool the Discriminator.

The Discriminator improves to detect fakes.

Over time, the Generator learns to produce realistic images.

How It Works (Simple Explanation)

Generator

Takes random noise as input.

Outputs a 28×28 fake image (like a handwritten digit).

Discriminator

Takes an image (real or fake).

Predicts whether it’s real (1) or fake (0).

Training Process

Step 1: Train Discriminator with real & fake images.

Step 2: Train Generator to fool the Discriminator.

Output

Initially, fake images look like noise.

After ~2000+ epochs, digits start to form (0–9 shapes


        
