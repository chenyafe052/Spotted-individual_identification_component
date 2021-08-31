# %% Import libraries
from keras import backend as K
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Sequential


# %% function: build_base_network()
def build_base_network(input_shape):
    seq = Sequential()

    nb_filter = [64, 128, 128, 256]

    # convolutional layer 1
    seq.add(Conv2D(nb_filter[0],
                   (10, 10),
                   input_shape=input_shape,
                   padding='valid',
                   data_format="channels_last"))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
    seq.add(Dropout(.25))

    # convolutional layer 2
    seq.add(Conv2D(nb_filter[1],
                   (7, 7),
                   input_shape=input_shape,
                   padding='valid',
                   data_format="channels_last"))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
    seq.add(Dropout(.25))

    # convolutional layer 3
    seq.add(Conv2D(nb_filter[2],
                   (4, 4),
                   input_shape=input_shape,
                   padding='valid',
                   data_format="channels_last"))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
    seq.add(Dropout(.25))

    # convolutional layer 4
    seq.add(Conv2D(nb_filter[3],
                   (4, 4),
                   input_shape=input_shape,
                   padding='valid',
                   data_format="channels_last"))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
    seq.add(Dropout(.25))

    # flatten
    seq.add(Flatten())
    seq.add(Dense(4096, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(4096, activation='relu'))

    return seq
    

# %% function: contrastive_loss()
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))