from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import Reshape, Conv2D, Flatten, BatchNormalization, LSTM, Activation
from keras.layers.advanced_activations import LeakyReLU

# Construct the Generator
def generator_model(noise_size, output_size):

    # Take the input noise
    G_input = Input(shape=(noise_size,))

    # Construct the Model
    G = Dense(32, kernel_initializer='glorot_uniform')(G_input)
    G = LeakyReLU(0.2)(G)
    G = Dense(64)(G)
    G = LeakyReLU(0.2)(G)

    G_output = Dense(output_size)(G)
    # G_output = Activation("tanh")(G_output)

    generator = Model(G_input, G_output)

    return generator

# Construct the Discriminator
def discriminator_model(GAN_size):

    # Take the generated output
    D_input = Input(shape=(GAN_size,))

    # Construct the Model
    D = Dense(256)(D_input)
    D = LeakyReLU(0.2)(D)
    D = Dense(128)(D)
    D = LeakyReLU(0.2)(D)
    D = Dense(32)(D)
    D = LeakyReLU(0.2)(D)
    # D = Dropout(0.2)(D)

    D_output = Dense(1, activation='sigmoid')(D)

    discriminator = Model(D_input, D_output)

    return discriminator

## CNN MODEL ## 

def generator_model_cnn(noise_size, output_size):

    G_input = Input(shape=(noise_size,))

    G = Dense(128, kernel_initializer='glorot_normal')(G_input)
    G = Activation('tanh')(G)
    G = BatchNormalization(momentum=0.99)(G)
    G = Reshape((32, 4))(G)
    G = LSTM(32, return_sequences=True)(G)
    G = LSTM(16, return_sequences=False)(G)
    G = Activation('tanh')(G)

    G_output = Dense(output_size, activation="tanh")(G)

    generator = Model(G_input, G_output)

    return generator

def discriminator_model_cnn(GAN_size):

    D_input = Input(shape=(GAN_size,))

    D = Dense(128)(D_input)
    D = Reshape((8, 8, 2))(D)
    D = Conv2D(64, kernel_size=3, strides=1, padding="same")(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Conv2D(32, kernel_size=3, strides=1, padding="same")(D)
    #D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Conv2D(16, kernel_size=3, strides=1, padding="same")(D)
    #D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Flatten()(D)
    #D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Dropout(0.2)(D)

    D_output = Dense(1, activation="sigmoid")(D)
    # D_output = Dense(1)(D)

    discriminator = Model(D_input, D_output)
    return discriminator