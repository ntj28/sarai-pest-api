import numpy
import cv2
import csv

#array of classes
species=[]
with open("pests.csv",'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        species.append(row[-1])

#2d array of features computed
data=[]
with open("pesttraining.csv",'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row[:-1])

#convert data to numpy array
inputs = numpy.empty( (len(data), len(data[0])), 'float' )
for i in range(len(data)):
  a = numpy.array(list(data[i]))
  f = a.astype('float')
  inputs[i,:]=f[:]

#2d array of classes the features correspond to (1 if it's their class)
targets= -1 * numpy.ones( (len(inputs), len(species)), 'float' )
i=0
with open("pesttraining.csv",'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        targets[i][species.index(row[-1])]=1
        i=i+1
print targets

ninputs = len(data[0])#number of features

# 8 hidden nodes.  If you change this to, for instance, 2, the network
# won't work well.
nhidden = 90 #60 130

# We should have one output for each input vector
noutput = len(species)#number of classes

# Create an array of desired layer sizes for the neural network
layers = numpy.array([ninputs, nhidden, noutput])

# Create the neural network
nnet = cv2.ANN_MLP(layers)

# Some parameters for learning.  Step size is the gradient step size
# for backpropogation.
step_size = 0.003 #0.01

# Momentum can be ignored for this example.
momentum = 0.10 #0.08

# Max steps of training
nsteps = 900 #5000

# Error threshold for halting training
max_err = 0.0000000001 #0.00001

# When to stop: whichever comes first, count or error
condition = cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS

# Tuple of termination criteria: first condition, then # steps, then
# error tolerance second and third things are ignored if not implied
# by condition
criteria = (condition, nsteps, max_err)

# params is a dictionary with relevant things for NNet training.
params = dict( term_crit = criteria,train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,bp_dw_scale = step_size, bp_moment_scale = momentum )

# Train our network
num_iter = nnet.train(inputs,targets,None,params=params)

# Create a matrix of predictions
predictions = np.empty_like(targets)

# See how the network did.
nnet.predict(inputs, predictions)

# Compute sum of squared errors
sse = numpy.sum( (targets - predictions)**2 )


# Compute # correct
true_labels = numpy.argmax( targets, axis=1 )
pred_labels = numpy.argmax( predictions, axis=1 )
num_correct = numpy.sum( true_labels == pred_labels )



print 'Predictions:'
for i in predictions:
  print species[numpy.argmax(i)]
print len(targets)
print true_labels
print pred_labels
print num_correct
print 'ran for %d iterations' % num_iter
print 'sum sq. err:', sse
print 'accuracy:', float(num_correct) / len(true_labels)

