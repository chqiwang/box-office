'''
This is a simple full-connected Neural Network. 
'''
__author__ = 'Chunqi Wang'
__date__ = 'November,2015'

import numpy as np
import numpy.random as rd
import numpy.linalg as la
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import itertools as it

def linear(x):
    return x
def d_linear(z):
    return 1.0
def sigmod(x):
    return 1.0 / (1.0 + np.exp(-x))
def d_sigmod(z):
    return z * (1.0 - z)
def tanh(x):
    return (np.exp(x) - 1.0) / (np.exp(x) + 1.0)
def d_tanh(z):
    return 1.0 / 2.0 * (1.0 + z) * (1.0 - z)
def softmax(x):
    return np.exp(x) / sum(np.exp(x))
def d_softmax(z):
    return z * (1.0 - z)
def mse(z, y):
    return z - y

input_len = 3
output_len = 3

hidden_depth = 1
hidden_unit_numbers = [16] * hidden_depth
hidden_activations = [tanh] * hidden_depth
hidden_activations_d = [d_tanh] * hidden_depth

output_activations = sigmod
output_activations_d = d_sigmod

mode = 'single'

iterations = 10**3 * 2
lamd = 0.1
eta = 0.001

show_info = 10**2

def NN(X, Y):
    layers = hidden_depth + 1 # Hidden plus output, while input are not included.
    layer_unit_numbers = hidden_unit_numbers + [output_len]
    layer_activations = hidden_activations + [output_activations]
    layer_activations_d = hidden_activations_d + [output_activations_d]

    # Alloc memories.
    W = [rd.randn(input_len, layer_unit_numbers[0])]
    for i in xrange(layers - 1):
        W.append(rd.randn(layer_unit_numbers[i], layer_unit_numbers[i + 1]))
    B = [rd.randn(i) for i in layer_unit_numbers]

    Z = [np.array(input_len)] + [np.array(i) for i in layer_unit_numbers]
    Delta = [np.array(i) for i in layer_unit_numbers]

    def output(x):
        Z[0] = x
        for i in xrange(layers):
            Z[i + 1] = layer_activations[i]((np.dot(Z[i], W[i]) + B[i]))
        return Z[-1]
    def predict(x):
        return np.argmax(output(x))

    if mode == 'batch':
        W_update = [np.zeros((input_len, layer_unit_numbers[0]))]
        for i in xrange(layers - 1):
            W_update.append(np.zeros((layer_unit_numbers[i], layer_unit_numbers[i + 1])))
        B_update = [np.zeros(i) for i in layer_unit_numbers]

    objvalues = []

    for iteration in xrange(iterations):
        if iteration % show_info == 0:
            e = np.sum([(output(x) - y)**2 for x, y in it.izip(X, Y)])
            print 'iteration %d. '%iteration + 'error %s.'%e
            objvalues.append(e)
        for x, y in it.izip(X, Y):
            # Forward.
            Z[0] = x
            for i in xrange(layers):
                Z[i + 1] = layer_activations[i]((np.dot(Z[i], W[i]) + B[i]))
            # Backward.
            Delta[-1] = layer_activations_d[-1](Z[-1]) * mse(Z[-1], y)
            for i in xrange(layers - 1, -1, -1):
                delta = Delta[i]
                if mode == 'single':
                    W[i] -= lamd * np.outer(Z[i], delta)
                    B[i] -= lamd * delta
                else:
                    W_update[i] -= lamd * np.outer(Z[i], delta)
                    B_update[i] -= lamd * delta
                if i > 0:
                    Delta[i - 1] = layer_activations_d[i - 1](Z[i]) * np.dot(W[i], delta)
        if mode == 'batch':
            for i in xrange(layers - 1, -1, -1):
                W[i] += W_update[i] - eta * W[i]
                B[i] += B_update[i] - eta * B[i]
                W_update[i] *= 0
                B_update[i] *= 0
    return predict, objvalues

def error_rate(X, Y, predictor):
    c = 0.0
    for x, y in  it.izip(X, Y):
        if y[predictor(x)] != 1:
            c += 1.0
    return c / len(X)