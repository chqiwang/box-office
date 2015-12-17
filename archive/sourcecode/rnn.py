'''
A simple RNN.
'''

__author__ = 'Chunqi Wang'
__date__ = 'November,2015'

from itertools import *
import random
import numpy as np
import numpy.linalg as la
import numpy.random as rd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

# Activation functions.
linear = lambda x: x
d_linear = lambda z: 1.0
sigmod = lambda x: 1.0 / (1.0 + np.exp(-x))
d_sigmod = lambda z: z * (1.0 - z)
tanh = lambda x: (np.exp(x) - 1.0) / (np.exp(x) + 1.0)
d_tanh = lambda z: 1.0 / 2.0 * (1.0 + z) * (1.0 - z)
softmax = lambda x: np.exp(x) / sum(np.exp(x))
d_softmax = lambda z: z * (1.0 - z)
# ReLu = lambda x: (x + np.abs(x)) / 2.0
# d_ReLu = lambda z: (z > 0).astype(np.float)

mse = lambda z, y: z - y

class RNN():
    def __init__(self, input_dim, ouput_dim, hidden_dim = 32, max_seq_len = 100, lamd = 0.0001, debug = False):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.ouput_dim = ouput_dim
        self.max_seq_len = max_seq_len

        self.debug = debug

        self.hidden_activation = tanh
        self.d_hidden_activation = d_tanh
        self.output_activation = linear
        self.d_output_activation = d_linear

        self.W_i = rd.randn(self.input_dim, self.hidden_dim)
        self.W_h = rd.randn(self.hidden_dim, self.hidden_dim)
        self.B_i = rd.randn(self.hidden_dim)
        self.W_o = rd.randn(self.hidden_dim, self.ouput_dim)
        self.B_o = rd.randn(self.ouput_dim)

        self.showinfo = 10**3
        self.iterations = 10**3*3
        self.lamd = lamd
        self.eta = 0.0001

    def fit(self, Seq_input, Seq_target):
        def generate_sequence():
            r = random.randint(0, len(Seq_input) - 1)
            return Seq_input[r], Seq_target[r]

        W_i_update = np.zeros_like(self.W_i)
        W_h_update = np.zeros_like(self.W_h)
        B_i_update = np.zeros_like(self.B_i)
        W_o_update = np.zeros_like(self.W_o)
        B_o_update = np.zeros_like(self.B_o)

        Z_h = np.zeros((self.max_seq_len + 1, self.hidden_dim))
        Z_o = np.zeros((self.max_seq_len, self.ouput_dim))
        Delta_h = np.zeros_like(Z_h)
        Delta_o = np.zeros_like(Z_o)

        Weights = [self.W_i, self.W_h, self.B_i, self.W_o, self.B_o]
        Weights_update = [W_i_update, W_h_update, B_i_update, W_o_update, B_o_update]
        temporary = [W_i_update, W_h_update, B_i_update, W_o_update, B_o_update, Z_h, Z_o, Delta_h, Delta_o]

        # Train the network by BP algorithm.
        for iteration in xrange(self.iterations):
            # Generate a random sample.
            input_seq, output_seq  = generate_sequence()
            seq_len = len(input_seq)
            # Forward.
            for i in xrange(seq_len):
                x = input_seq[i]
                Z_h[i] = self.hidden_activation(x.dot(self.W_i) + Z_h[i - 1].dot(self.W_h) + self.B_i)
                Z_o[i] = self.output_activation(Z_h[i].dot(self.W_o) + self.B_o)
            # Backward.
            for i in xrange(seq_len - 1, -1, -1):
                x, y = input_seq[i], output_seq[0]
                Delta_o[i] = mse(Z_o[i], y) * self.d_output_activation(Z_o[i])
                Delta_h[i] = (Delta_o[i].dot(self.W_o.T) + Delta_h[i + 1].dot(self.W_h.T)) * self.d_hidden_activation(Z_h[i])
                # Calculate updates.
                W_o_update -= np.outer(Z_h[i], Delta_o[i])
                B_o_update -= Delta_o[i]
                W_i_update -= np.outer(x, Delta_h[i])
                B_i_update -= Delta_h[i]
                W_h_update -= np.outer(Z_h[i - 1], Delta_h[i])
            
            if self.debug:
                # Gradient checking.
                g = self.gradient(input_seq, output_seq, self.W_h[0:1, 0:1])
                if abs(g + W_h_update[0, 0]) >= 10**(-5):
                    print 'gradient check error.', g, - W_h_update[0, 0]
                else:
                    print 'gradient check right.', g, - W_h_update[0, 0]
                
            # Update all weights.
            for w, w_u in izip(Weights, Weights_update):
                w += self.lamd * w_u - self.eta * w
            # Reset memories for reuse.
            for t in temporary: t *= 0

            if iteration % self.showinfo == 0:
                print 'iterations: ',iteration , ' object function value: ', self.object_function_value(input_seq, output_seq)

    def predict(self, Seq_input):
        result = []
        pre_z_h = np.zeros(self.hidden_dim)
        for input_seq in Seq_input:
            seq_len = len(input_seq)
            Z_o = []
            for i in xrange(seq_len):
                x = input_seq[i]
                z_h = self.hidden_activation(x.dot(self.W_i) + pre_z_h.dot(self.W_h) + self.B_i)
                Z_o.append(self.output_activation(z_h.dot(self.W_o) + self.B_o))
            result.append(Z_o)
        return np.array(result)

    def object_function_value(self, input_seq, output_seq):
        seq_len = len(input_seq)
        pre_z_h = np.zeros(self.hidden_dim)
        error = 0
        for i in xrange(seq_len):
            x = input_seq[i]
            z_h = self.hidden_activation(x.dot(self.W_i) + pre_z_h.dot(self.W_h) + self.B_i)
            z_o = self.output_activation(z_h.dot(self.W_o) + self.B_o)
            pre_z_h = z_h
            error += (z_o - output_seq[0])[0]**2
        return error / 2.0

    def gradient(self, input_seq, output_seq, w):
        v1 = self.object_function_value(input_seq, output_seq)
        delta = 10**(-5)
        w += delta
        v2 = self.object_function_value(input_seq, output_seq)
        w -= delta
        return (v2 - v1) / delta