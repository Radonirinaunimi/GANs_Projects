import numpy as np 
from random import sample

from keras import Model
from keras.layers import Input
from keras.optimizers import Adam, RMSprop, Adadelta, SGD

from model import generator_model, discriminator_model
from model import generator_model_cnn, discriminator_model_cnn
from ploting import plot_generated_pdf
from pdformat import x_pdf, sample_pdf

# Generate the data
pdf_dataX = sample_pdf(len(x_pdf))
length = pdf_dataX.shape[0]*pdf_dataX.shape[1]
pdf_data = pdf_dataX.reshape(length,)


# Define parameters
random_noise_dim = 100
learning_rate = 0.01
optimizer1 = SGD(lr=learning_rate)
optimizer2 = SGD(lr=learning_rate)

# Call the Generator Model
def make_generator():
    return generator_model(random_noise_dim, length)

# Call the Discriminator Model
def make_discriminator():
    return discriminator_model(length)

# Compile the Gen
Generator = make_generator()
Generator.name = "generator"
Generator.compile(loss='mean_squared_error', optimizer = optimizer1)
Generator.summary()

# Compile the Dis
Discriminator = make_discriminator()
Discriminator.name = 'discriminator'
Discriminator.compile(loss='binary_crossentropy', optimizer=optimizer2, metrics=['accuracy'])
Discriminator.summary()

# Choose to train the Discriminator or Not
always_train_Discriminator = False

Gan_input  = Input(shape = (random_noise_dim,))
Gan_latent = Generator(Gan_input)
Gan_output = Discriminator(Gan_latent)

if not always_train_Discriminator:
    Discriminator.trainable = False
GAN = Model(inputs = Gan_input, outputs = Gan_output)
GAN.name   = "gan"
GAN.compile(loss='binary_crossentropy', optimizer = optimizer1)
GAN.summary()

# Set the number of training 
number_training = 15000
batch_size = 1

# Number of steps to train G&D
nd_steps = 4
ng_steps = 6

f = open('loss.csv','w')
f.write('Iteration,Discriminator Loss,Discriminator accuracy,Generator Loss\n')

# Train the Model
for k in range(1,number_training+1):
    noise = np.random.normal(0,1,size=[batch_size,random_noise_dim])
    pdf_batch = [pdf_data] # put 10 members here so you compare 10 trues, 10 fakes each time
    pdf_fake  = Generator.predict(noise)

    xinput = np.concatenate([pdf_batch,pdf_fake])

    for m in range(nd_steps): 
        # Train the Discriminator with trues and fakes
        y_Discriminator = np.zeros(2*batch_size)
        y_Discriminator[:batch_size] = 1.0
        if not always_train_Discriminator:
            Discriminator.trainable = True  # Ensure the Discriminator is trainable
        dloss = Discriminator.train_on_batch(xinput, y_Discriminator)

    for n in range(ng_steps):
        # Train the generator by generating fakes and lying to the Discriminator
        noise = np.random.normal(0, 1, size = [batch_size, random_noise_dim])
        y_gen = np.ones(batch_size)                                   
        if not always_train_Discriminator:
            Discriminator.trainable = False
        gloss = GAN.train_on_batch(noise, y_gen)

    loss_info = "Iteration %d: \t .D loss: %f \t D acc: %f" % (k, dloss[0], dloss[1])
    loss_info = "%s  \t .G loss: %f" % (loss_info, gloss)

    if k % 100 == 0:
        print(loss_info)
        f.write("%d,%f,%f,%f\n"%(k,dloss[0],dloss[1],gloss))

    if k % 1000 ==0:
        print("Training {0}".format(k))
        plot_generated_pdf(x_pdf, pdf_dataX, random_noise_dim, k,Generator, always_train_Discriminator)
f.close()
