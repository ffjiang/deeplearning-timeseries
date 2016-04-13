import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class LSTMCell: 

    # Size is the dimensionality of the input vector
    def __init__(self, size):
        self.W_f = np.zeros((size, size)) # Forget gate matrix (for input)
        self.W_i = np.zeros((size, size)) # Input gate matrix (for input)
        self.W_c = np.zeros((size, size)) # Candidate value matrix (for input)
        self.W_o = np.zeros((size, size)) # Output gate matrix (for input)

        self.U_f = np.zeros((size, size)) # Forget gate matrix (for prev output)
        self.U_i = np.zeros((size, size)) # Input gate matrix (for prev output)
        self.U_c = np.zeros((size, size)) # Candidate value matrix (for prev output)
        self.U_o = np.zeros((size, size)) # Output gate matrix (for prev output)

        self.b_f = np.zeros(size) # Forget gate bias
        self.b_i = np.zeros(size) # Input gate bias
        self.b_c = np.zeros(size) # Candidate value bias
        self.b_o = np.zeros(size) # Output gate bias

        self.h = np.zeros(size) # output at current time
        self.C = np.zeros(size) # state at current time

    # x is the input vector, returns output h
    def forwardPass(self, x):
        # Compute forget gate vector
        f = sigmoid(np.dot(self.W_f, x) + np.dot(self.U_f, self.h) + self.b_f))

        # Compute input gate vector
        i = sigmoid(np.dot(self.W_i, x) + np.dot(self.U_i, self.h) + self.b_i))

        # Compute the candidate value vector
        C-bar = np.tanh(np.dot(self.W_c, x) + np.dot(self.U_c, self.h) + self.b_c))

        # Compute the new state vector as the elements of the old state allowed
        # through by the forget gate, plus the candidate values allowed through
        # by the input gate
        self.C = np.dot(f_t, self.C) + np.dot(i, C-bar)

        # Compute the output gate vector
        o = sigmoid(np.dot(self.W_o, x) + np.dot(self.U_o, self.h) + self.b_o))

        # Compute the new output
        self.h = np.dot(o, np.tanh(self.C))

        return self.h

    # Back propagation through time
    def BPTT(self):

class LSTMNetwork:

    # Structure is a vector specifing the structure of the network - each
    # element represents the number of nodes in that layer
    def __init__(self, structure):
        self.layers = [[x for x in structure]] # this doesnt make sense
        for 
