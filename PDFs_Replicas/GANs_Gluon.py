# Import the Libraries
import lhapdf
import tensorflow as tf
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
sb.set_style("whitegrid")

# Get the PDF4LHC15 for test purpose and print some description
pdf = lhapdf.getPDFSet("NNPDF31_nnlo_as_0118")
print(pdf.description)
# Take only the central value
pdf_central = pdf.mkPDF(0)

# Define the scale 
Q_pdf = 125

# Define a function which does the sampling
def sample_pdf(n=1000):
    data  = []
    x_pdf = []
    for i in np.logspace(-3,-1,n/2):
        x_pdf.append(i)
    for i in np.linspace(0.1,1,n/2):
        x_pdf.append(i)
    for x in x_pdf:
        y_pdf = pdf_central.xfxQ(21,x,Q_pdf)
        data.append([x,y_pdf])
    return np.array(data)

# Define the function which samples the Data
def sample_noise(m, n):
    return np.random.uniform(0, 1., size=[m, n])

# Implement the GENERATOR Model

# The following function takes the follwoing as input:
# A Random Sample Z
# A list which contains the Structure of the NN (layers)
# A variable called "reuse"
def generator(Z, layers_size=[16, 16],reuse=False):
    # Create and Share a variable named "Generator"
    # Here "reuse" allows us to share the variable
    with tf.variable_scope("Generator",reuse=reuse):
        # Define the 1st and 2nd layer with "leaky_relu" as an activation fucnction
        l1 = tf.layers.dense(Z,layers_size[0],activation=tf.nn.leaky_relu)
        l2 = tf.layers.dense(l1,layers_size[1],activation=tf.nn.leaky_relu)
        # Define the output layer with 2 nodes
        # This dimension is correspond to the dimension of the "real dataset"
        output = tf.layers.dense(l2,2)

    return output

# Implement the DISCRIMINATOR Model

# The following function takes the following as input:
# A sample from the "REAL" dataset
# A list which contains the structure of the first 2 "hidden layer"
# A variable called "reuse"
def discriminator(X,layers_size=[16, 16],reuse=False):
    # Create and share a variable named "Discriminator"
    with tf.variable_scope("Discriminator",reuse=reuse):
        # Define the 1st and 2nd layer with "leaky_relu" as an activation fucnction
        l1 = tf.layers.dense(X,layers_size[0],activation=tf.nn.leaky_relu)
        l2 = tf.layers.dense(l1,layers_size[1],activation=tf.nn.leaky_relu)
        # Fix the third layer to 2 nodes so we can visualize the transformed feature space in a 2D plane
        l3 = tf.layers.dense(l2,2)
        # Define the output layer (logit)
        output = tf.layers.dense(l3,1)

    return output, l3

# Adversarial Training

# Initialize the placeholder for the real sample
X = tf.placeholder(tf.float32,[None,2])
# Initialize the placeholder for the random sample
Z = tf.placeholder(tf.float32,[None,2])

# Define the Graph which Generate fake data from the Generator and feed the Discriminator
G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample,reuse=True)

# Define the Loss Function for G and D
DiscriminatorLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) +
                           tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
GeneratorLoss     = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))

# Train the NN based on the Model defined above
GeneratorVars     = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Generator")
DiscriminatorVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Discriminator")

# Define the Miniminzer for G&D
GeneratorStep     = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(GeneratorLoss,var_list = GeneratorVars)
DiscriminatorStep = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(DiscriminatorLoss,var_list = DiscriminatorVars)

# sess = tf.Session(config=config)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

batch_size = 256
nd_steps   = 10
ng_steps   = 10

x_plot = sample_pdf(n=batch_size)

f = open('loss.csv','w')
f.write('Iteration,Discriminator Loss,Generator Loss\n')

numb_training = 10001

for i in range(numb_training):
    X_batch = sample_pdf(n=batch_size)
    Z_batch = sample_noise(batch_size, 2)

    for _ in range(nd_steps):
        _, dloss = sess.run([DiscriminatorStep, DiscriminatorLoss], feed_dict={X: X_batch, Z: Z_batch})
    rrep_dstep, grep_dstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

    for _ in range(ng_steps):
        _, gloss = sess.run([GeneratorStep, GeneratorLoss], feed_dict={Z: Z_batch})

    rrep_gstep, grep_gstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

    print ("Iterations: %d\t out of %d\t. Discriminator loss: %.4f\t Generator loss: %.4f"%(i,numb_training-1,dloss,gloss))
    if i%100 == 0:
        f.write("%d,%f,%f\n"%(i,dloss,gloss))

    if i%1000 == 0:
        plt.figure()
        g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})
        xax = plt.scatter(x_plot[:,0], x_plot[:,1])
        gax = plt.scatter(g_plot[:,0],g_plot[:,1])

        plt.legend((xax,gax), ("Real Data","Generated Data"))
        plt.title('Samples at Iteration %d'%i)
        plt.tight_layout()
        plt.savefig('plots/iterations/iteration_%d.png'%i, dpi=250)
        plt.close()

f.close()