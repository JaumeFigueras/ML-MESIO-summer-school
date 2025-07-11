################################################################################
############################### INITIALIZE #####################################
################################################################################
import numpy as np
import random

""" Read data:
Any of the 3 sets (train, valid, test) comes as a 2 column matrix:

- 1st: (train[0][i]) contains for each i a vector of 784 values
(float32) corresponding to a 28x28 pixel intensity (grey scale)

- 2nd: (train[1][i]) contains the correct digit (int64) that represents the corresponding image"""

import pickle
import gzip
f = gzip.open('./BBDD Input/mnist.pkl.gz', 'rb')
train, valid, test = pickle.load(f,encoding='latin1')
f.close()

# Print some numbers:
import matplotlib.cm as cm
import matplotlib.pyplot as plt
for i in range(0,10):
    plt.figure()
    plt.imshow(train[0][i].reshape((28, 28)), cmap=cm.Greys_r)
    plt.title('REAL: ' + str(train[1][i]), fontsize=20)
    plt.show()

def vectorized_result(j):
    """This function outputs a column vector of lenagth 10 with all 0s and a 1 in j position. We have converted a 0,
    1,2,...,9 digit to a Neural Network output"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# Each image comes as a row vector, convert it as a column to be a valid input for the NN
X_train = [np.reshape(x, (784, 1)) for x in train[0]]

# Change digit to column vector
Y_train = [vectorized_result(y) for y in train[1]]

# Final train set:
train = list(zip(X_train, Y_train))

print(train[0])


# Convert validation images (the digit is not necessary as we don't put it to feed a NN)
X_valid = [np.reshape(x, (784, 1)) for x in valid[0]]
Y_valid=valid[1]
valid = list(zip(X_valid, Y_valid))

# And test:
X_test = [np.reshape(x, (784, 1)) for x in test[0]]
Y_test = test[1]
test = list(zip(X_test, Y_test))


################################################################################
################################ NN Class ######################################
################################################################################

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j,
                self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


################################################################################
############################# Train Network ####################################
################################################################################

# Define network architecture
net = Network([784, 100, 10])

# Train network
net.SGD(training_data=train,
        epochs=30,
        mini_batch_size=10,
        eta=3.0,
        test_data=test)


test_results = [(np.argmax(net.feedforward(x)), y) for (x, y) in test]
net.evaluate(test_data=test)

import matplotlib.cm as cm
import matplotlib.pyplot as plt

for i in range(20):
    pred=net.feedforward(test[i][0])
    digit_pred=np.argmax(pred)
    prob_pred=pred[digit_pred]
    plt.figure()
    plt.imshow(test[i][0].reshape((28, 28)), cmap=cm.Greys_r)
    plt.title('Real / Predict: ' + str(test[i][1]) + ' / ' + str(digit_pred)
              + '\nProb: ' + str(round(prob_pred[0]*100, 2))+'%', fontsize=15)
    plt.show()

# Predictions for observation 8 and 18
n = 0
for i in net.feedforward(test[8][0]):
    print(n, ': ', round(i[0]*100, 4), '%', sep='')
    n += 1

n = 0
for i in net.feedforward(test[18][0]):
    print(n, ': ', round(i[0]*100, 4), '%', sep='')
    n += 1
