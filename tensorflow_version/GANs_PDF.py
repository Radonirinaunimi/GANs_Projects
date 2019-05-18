# Import the Libraries
import lhapdf
import math
import random
import tensorflow as tf
# import tensorflow.contrib.slim as slim
import numpy as np
import seaborn as sb
from random import sample
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
sb.set_style("whitegrid")

# Get the PDF4LHC15 for test purpose and print some description
pdf = lhapdf.getPDFSet("NNPDF31_nnlo_as_0118")
print(pdf.description)
# Take only the central value
pdf_init = pdf.mkPDFs()
pdf_central = sample(pdf_init,25)
size_member = len(pdf_central)

# Define the scale 
Q_pdf = 1.7874388

# Nodes from LHAPDF
x_nodes = "1.0000000e-09 1.2805087e-09 1.6397027e-09 2.0996536e-09 2.6886248e-09 3.4428076e-09 4.4085452e-09 5.6451808e-09 7.2287034e-09 9.2564179e-09 1.1852924e-08 1.5177773e-08 1.9435271e-08 2.4887035e-08 3.1868066e-08 4.0807337e-08 5.2254152e-08 6.6911899e-08 8.5681272e-08 1.0971562e-07 1.4049181e-07 1.7990099e-07 2.3036479e-07 2.9498413e-07 3.7772976e-07 4.8368627e-07 6.1936450e-07 7.9310166e-07 1.0155736e-06 1.3004509e-06 1.6652388e-06 2.1323528e-06 2.7304964e-06 3.4964246e-06 4.4772022e-06 5.7330966e-06 7.3412804e-06 9.4005738e-06 1.2037517e-05 1.5414146e-05 1.9737949e-05 2.5274616e-05 3.2364367e-05 4.1442855e-05 5.3067938e-05 6.7953959e-05 8.7015639e-05 1.1142429e-04 1.4267978e-04 1.8270270e-04 2.3395241e-04 2.9957810e-04 3.8361238e-04 4.9121901e-04 6.2901024e-04 8.0545312e-04 1.0313898e-03 1.3207036e-03 1.6911725e-03 2.1655612e-03 2.7730201e-03 3.5508765e-03 4.5469285e-03 5.8223817e-03 7.4556107e-03 9.5469747e-03 1.2224985e-02 1.5654200e-02 2.0045340e-02 2.5668233e-02 3.2868397e-02 4.2088270e-02 5.3894398e-02 6.9012248e-02 8.8370787e-02 1.0000000e-01 1.1216216e-01 1.2432432e-01 1.3648649e-01 1.4864865e-01 1.6081081e-01 1.7297297e-01 1.8513514e-01 1.9729730e-01 2.0945946e-01 2.2162162e-01 2.3378378e-01 2.4594595e-01 2.5810811e-01 2.7027027e-01 2.8243243e-01 2.9459459e-01 3.0675676e-01 3.1891892e-01 3.3108108e-01 3.4324324e-01 3.5540541e-01 3.6756757e-01 3.7972973e-01 3.9189189e-01 4.0405405e-01 4.1621622e-01 4.2837838e-01 4.4054054e-01 4.5270270e-01 4.6486486e-01 4.7702703e-01 4.8918919e-01 5.0135135e-01 5.1351351e-01 5.2567568e-01 5.3783784e-01 5.5000000e-01 5.6216216e-01 5.7432432e-01 5.8648649e-01 5.9864865e-01 6.1081081e-01 6.2297297e-01 6.3513514e-01 6.4729730e-01 6.5945946e-01 6.7162162e-01 6.8378378e-01 6.9594595e-01 7.0810811e-01 7.2027027e-01 7.3243243e-01 7.4459459e-01 7.5675676e-01 7.6891892e-01 7.8108108e-01 7.9324324e-01 8.0540541e-01 8.1756757e-01 8.2972973e-01 8.4189189e-01 8.5405405e-01 8.6621622e-01 8.7837838e-01 8.9054054e-01 9.0270270e-01 9.1486486e-01 9.2702703e-01 9.3918919e-01 9.5135135e-01 9.6351351e-01 9.7567568e-01 9.8783784e-01 1.0000000e+00"

# Construc the array of x values
sx_nodes = x_nodes.split()
x_pdf = np.array([float(x) for x in sx_nodes])

# Define a log uniform function
def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))

# Define a function which does the sampling
def sample_pdf(n=150):
    data  = []
    # x_pdf = []
    # m = math.floor(n/3)
    # # Take m random values between 1e-3 and 1e-1 in logscale
    # for i in loguniform(low=-3,high=-1,size=m): x_pdf.append(i)
    # # Taake (n-m) values between 0.1 and 1 
    # for i in np.random.uniform(0.1,1,n-m): x_pdf.append(i)
    # Construct the sampling
    flavors_list = [1,2,3,21]
    for p in pdf_central:
        repl = []
        for x in x_pdf:
            row = [x]
            for fl in flavors_list:
                if (fl<3): row.append(p.xfxQ2(fl,x,Q_pdf)-p.xfxQ2(-fl,x,Q_pdf))
                else : row.append(p.xfxQ2(fl,x,Q_pdf)/3)
            repl.append(row)
        data.append(repl)
    return np.array(data)

# Define the function which generates the noise data
# The shape of the input noise does not have to be same as the true pdf inputs
# It can be a 1-dimensional vector (but this does not perform well as the below)
def sample_noise(m, n):
    return np.random.uniform(0,1.,size=[m, n])

# Define a function which sorts a multi-dimensional list wtr to the 1st row
def sort_wrt_row(lst):
    new_lst = []
    ordering = lst[0].argsort()
    for i in range(len(lst)):
        new_lst.append(lst[i][ordering])
    return new_lst

# Sort with respect to the 1s column
def sort_wrt_col(lst):
    return lst[lst[:,0].argsort()]     

# Define the hidden layers 
nb_points = len(x_pdf)
hidden_gen = [16,32]
hidden_dis = [32,16]
# Take the data shape
data_shape = sample_pdf(n=nb_points).shape[2]

# Implement the GENERATOR Model

# The following function takes the follwoing as input:
# A Random Sample Z
# A list which contains the Structure of the NN (layers)
# A variable called "reuse"
def generator(input_noise, layers_size=hidden_gen,reuse=False):
    # Create and Share a variable named "Generator"
    # Here "reuse" allows us to share the variable
    with tf.variable_scope("Generator",reuse=reuse):
        # Define the 1st and 2nd layer with "leaky_relu" as an activation fucnction
        L1 = tf.layers.dense(input_noise,layers_size[0],activation=tf.nn.leaky_relu)
        L2 = tf.layers.dense(L1,layers_size[1],activation=tf.nn.leaky_relu)
        # Define the output layer with data_shape nodes
        # This dimension is correspond to the dimension of the "real dataset"
        output = tf.layers.dense(L2,data_shape)
    return output

# Implement the DISCRIMINATOR Model

# The following function takes the following as input:
# A sample from the "REAL" dataset
# A list which contains the structure of the first 2 "hidden layer"
# A variable called "reuse"
def discriminator(input_true,layers_size=hidden_dis,reuse=False):
    # Create and share a variable named "Discriminator"
    with tf.variable_scope("Discriminator",reuse=reuse):
        # Define the 1st and 2nd layer with "leaky_relu" as an activation fucnction
        L1 = tf.layers.dense(input_true,layers_size[0],activation=tf.nn.leaky_relu)
        L2 = tf.layers.dense(L1,layers_size[1],activation=tf.nn.leaky_relu)
        # Fix the third layer to 2 nodes so we can visualize the transformed feature space in a 2D plane
        L3 = tf.layers.dense(L2,data_shape)
        # Define the output layer (logit)
        output = tf.layers.dense(L3,1)
    return output, L3

# Adversarial Training

# Initialize the placeholder for the real sample
X = tf.placeholder(tf.float32,[None,nb_points,data_shape])
# Initialize the placeholder for the random sample
Z = tf.placeholder(tf.float32,[None,nb_points,data_shape])

# Define the Graph which Generate fake data from the Generator and feed the Discriminator
G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample,reuse=True)

# Define the Loss Function for G and D
DiscriminatorLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) +
                           tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
GeneratorLoss     = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))

# Collect the variables in the graph
GeneratorVars     = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Generator")
DiscriminatorVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Discriminator")

# Define the Optimizer for G&D
GeneratorStep     = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(GeneratorLoss,var_list = GeneratorVars)
DiscriminatorStep = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(DiscriminatorLoss,var_list = DiscriminatorVars)

# def model_summary():
#     model_vars = tf.trainable_variables()
#     slim.model_analyzer.analyze_vars(model_vars, print_info=True)

# model_summary()

# sess = tf.Session(config=config)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

nd_steps   = 10
ng_steps   = 10

# Fetch the real PDF in order to plot them
x_plot  = sample_pdf(n=nb_points)
# sx_plot = [sort_wrt_col(xp) for xp in x_plot]

f = open('loss.csv','w')
f.write('Iteration,Discriminator Loss,Generator Loss\n')

# Plot pdfs for every x trainings
# Generate random colors
col1 = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(data_shape)]
col2 = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(data_shape)]
def plot_generated_pdf(generator, noise, iteration, replicas, shape_data=data_shape, s=14, a=0.95):
    g_plot  = sess.run(generator, feed_dict={Z: noise})
    # sg_plot = [sort_wrt_col(xg) for xg in g_plot]

    plt.figure()
    for r in range(replicas):
        for gen in range(1,shape_data):
            plt.scatter(x_plot[r][:,0],x_plot[r][:,gen],color=col1[gen],s=14,alpha=a)
            plt.scatter(g_plot[r][:,0],g_plot[r][:,gen],color=col2[gen],s=14,alpha=a)
    plt.title('Samples at Iteration %d'%iteration)
    plt.xlim([0.001,1])
    plt.ylim([0,0.8])
    plt.tight_layout()
    plt.savefig('iterations/iteration_%d.png'%iteration, dpi=250)
    plt.close()

numb_training = 10001
batch_size  = 5
batch_count = int(sample_pdf().shape[0] / batch_size)

# Training
for i in range(numb_training):
    X_batch = sample_pdf(n=nb_points)[np.random.randint(0, sample_pdf().shape[0], size = batch_size)]
    Z_batch = [sample_noise(nb_points,data_shape) for i in range(batch_size)]

    # Train independently G&D in multiple steps
    for _ in range(nd_steps):
        _, dloss = sess.run([DiscriminatorStep, DiscriminatorLoss], feed_dict={X: X_batch, Z: Z_batch})

    for _ in range(ng_steps):
        _, gloss = sess.run([GeneratorStep, GeneratorLoss], feed_dict={Z: Z_batch})

    print ("Iterations: %d\t out of %d\t. Discriminator loss: %.4f\t Generator loss: %.4f"%(i,numb_training-1,dloss,gloss))
    if i%100 == 0:
        f.write("%d,%f,%f\n"%(i,dloss,gloss))

    # Plot each 1000 iteration
    if i%1000 == 0:
        plot_generated_pdf(G_sample, Z_batch, i, 5, shape_data=data_shape)

f.close()