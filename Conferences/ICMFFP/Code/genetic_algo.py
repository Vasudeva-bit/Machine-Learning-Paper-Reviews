from multiprocessing.dummy import active_children
import lidar_vae
import front_vae
import rear_vae
import left_vae
import right_vae
import tensorflow as tf
from tf.keras import layers, Sequential
from tf.keras.initializers import he_normal
import numpy as np

latent_dim = 100
lidar_frames = None # based on the lidar
tot_frames = lidar_frames+4

def mutation(weights, mutation_rate=1.4):
    for div, weight in weights:
        da=(np.random.rand(weight.shape[0]+1)-0.5)*0.1*mutation_rate
        weight += da*np.maximum(np.exp(-0.1*np.abs(weight)),-np.sign(weight*da))
        weights[div] = weight
    return weights

def get_weights():
    pass

def initailize_weights():
    weights = {}
    weight = np.array()
    weights.update({'0': weight})
    div = 2
    while((tot_frames*latent_dim)//div < 512):
        pass
    weights.update({str(div+2):None})
    weights.update({str(div+2):None})
    weights.update({str(div+2):None})
    weights.update({str(div+2):None})
    weights.update({str(div+2):None})
    weights.update({str(div+2):None})
    return weights

def genetic_algorithm(weights):
    model = Sequential()
    model.add(layers.Dense(tot_frames*latent_dim),
              input_shape=(tot_frames*latent_dim, 0),
              kernel_initializer=weights['0'],
              bias_initializer=he_normal())
    div = 2
    while((tot_frames*latent_dim)//div < 512):
        model.add(layers.Dense(tot_frames*latent_dim//div), activation='relu',
              kernel_initializer=weights[str(div)],
              bias_initializer=he_normal())
        div += 2
    model.add(layers.Dense(512), activation='relu',
              kernel_initializer=weights[str(div)],
              bias_initializer=he_normal())
    div+=2
    model.add(layers.Dense(256), activation='relu',
              kernel_initializer=weights[str(div)],
              bias_initializer=he_normal())
    div+=2
    model.add(layers.Dense(128), activation='relu',
              kernel_initializer=weights[str(div)],
              bias_initializer=he_normal())
    div+=2
    model.add(layers.Dense(64), activation='relu',
              kernel_initializer=weights[str(div)],
              bias_initializer=he_normal())
    div+=2
    model.add(layers.Dense(16), activation='relu',
              kernel_initializer=weights[str(div)],
              bias_initializer=he_normal())
    div+=2
    model.add(layers.Dense(3), activation='relu',
              kernel_initializer=weights[str(div)],
              bias_initializer=he_normal())
    # Note: the bias_initializer can also be assigned to custom bias weights but, to make the idea simple they are conidered of he_normal
    return model

def natural_selection(generations=100):
    weights = initailize_weights()
    model = genetic_algorithm(weights)
    for gen in range(generations):
        

