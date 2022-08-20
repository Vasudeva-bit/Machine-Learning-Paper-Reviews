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
    for _ in range(num_epochs):
        for _ in range(len(train_data)//batch_size):
            (train_image) = next(train_image_data.as_numpy_iterator())
            if(mode == 0):
                loss = vae.vae_training_step(train_image)
                loss_history.append(loss)
            else:
                latent_space = vae.encode(train_image)
    if(not(mode == 0)):
        return np.array(latent_space)
