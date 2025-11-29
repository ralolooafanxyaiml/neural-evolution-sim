import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Conv2D, Conv2DTranspose, Flatten, LeakyReLU, Dropout, Input
from tensorflow.keras.optimizers import Adam
import os
import cv2

LATENT_DIM = 100
IMG_SHAPE = (64, 64, 3)
EPOCHS = 2000
BATCH_SIZE = 32
DATA_DIR = "./visual_datasets"

def load_real_images():
    data = []

    if not os.path.exists(DATA_DIR):
        print("ERROR!)
        return np.array([])

    valid_extension = {".jpg", ".jpeg", ".png"}
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                try:
                    img_path = os.path.join(root,file)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (64, 64))
                    img = (img - 127.5) / 127.5
                    data.append(img)
                except Exception as e:
                    print("ERROR")
    if len(data) == 0:
        print("ERROR!")
        return np.array([])

    print("Finished.")
    return np.array(data)

def build_generator():
    model = Sequential()
    n_nodes = 8 * 8 * 128
    model.add(Dense(n_nodes, input_dim=LATENT_DIM))
    model.add(LeakyReLU(alpha=0.2))
    model.add(ReShape((8, 8, 128)))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2,2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2,2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(3, (4, 4), strides=(2,2), padding="same", activation="tanh"))
    return model

def build_discriminator():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding="same", input_shape=IMG_SHAPE))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="sigmoid"))
  
    optimizer1 = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=optimizer1, metrics=["accuracy"])
    return model

def train():
    dataset = load_real_images()
    if len(dataset) == 0: return

    generator = build_generator()
    discriminator = build_discriminator()

    discriminator.trainable = False
    gan_input = Input(shape=(LATENT_DIM,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))

    real = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))
 
    for epoch in range(EPOCHS):
        idx = np.random.randint(0, dataset.shape[0], BATCH_SIZE)
        imgs = dataset[idx]

        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        gen_imgs = generator.predict(noise, verbose=0)
  
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
   
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        g_loss = gan.train_on_batch(noise, real)

    print("Finished")
    generator.save("generator_model.h5")

if __name__ == "__main__":
    train()
       