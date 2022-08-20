import tensorflow as tf
import functools
import numpy as np
import sys

import VAE as vae

# get the mode of use, train or deploy
mode = int(sys.argv[1])

# collect the data from the lidar sensor
if(mode == 0): # mode 0 is train
    train_data = None # folder path
else: # mode other than 0 is deploy
    train_data = None # open CV VideoCapture() from LIDAR

train_image_data = None  # get the tensor.slices

# global parameters
batch_size = 64
learning_rate = 1e-4
latent_dim = 100
num_epochs = 20
optimizer_vae = tf.keras.optimizers.Adam(learning_rate)
vae = VAE(latent_dim)

def fit(self):
    loss_history = []
    latent_space = []
    for _ in range(num_epochs):
        for _ in range(len(train_data)//batch_size):
            (train_images) = next(train_image_data.as_numpy_iterator())
            for train_image in train_images:
                if(mode == 0):
                    loss = vae.vae_training_step(train_image)
                else:
                    latent = vae.encode(train_image)
                    latent_space.append(latent)
            if(mode == 0):
                loss_history.append(loss/len(train_images))
    if(not(mode == 0)):
        return np.array(latent_space)
