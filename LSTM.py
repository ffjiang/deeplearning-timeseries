import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class LSTMCell: 

    # Size is the dimensionality of the input vector
    def __init__(self, size, numCells):
        self.size = size
        self.numCells = numCells

        self.W_f = np.zeros((numCells, size)) # Forget gate matrix (for input)
        self.W_i = np.zeros((numCells, size)) # Input gate matrix (for input)
        self.W_c = np.zeros((numCells, size)) # Candidate value matrix (for input)
        self.W_o = np.zeros((numCells, size)) # Output gate matrix (for input)

        self.U_f = np.zeros((numCells, numCells)) # Forget gate matrix (for prev output)
        self.U_i = np.zeros((numCells, numCells)) # Input gate matrix (for prev output)
        self.U_c = np.zeros((numCells, numCells)) # Candidate value matrix (for prev output)
        self.U_o = np.zeros((numCells, numCells)) # Output gate matrix (for prev output)

        self.h = [] # output at time t
        self.C = [] # state at time t

        self.C_bar = []
        self.i = []
        self.f = []
        self.o = []

        self.I = []
        self.z = []

    # x is the input vector (including bias term), returns output h
    def forwardStep(self, x):
        I = np.concatenate((x, self.h[-1]))
        self.I.append(I)

        z = np.dot(self.W, I)
        self.z.append(z)

        # Compute the candidate value vector
        C_bar = np.tanh(z[0:self.numCells])
        self.C_bar.append(C_bar)

        # Compute input gate vector
        i = sigmoid(z[self.numCells:self.numCells * 2])
        self.i.append(i)

        # Compute forget gate vector
        f = sigmoid(z[self.numCells * 2:self.numCells * 3])
        self.f.append(f)

        # Compute the output gate vector
        o = sigmoid(z[self.numCells * 3:])
        self.o.append(o)

        # Compute the new state vector as the elements of the old state allowed
        # through by the forget gate, plus the candidate values allowed through
        # by the input gate
        C = np.dot(f, self.C[-1]) + np.dot(i, C_bar)
        self.C.append(C)

        # Compute the new output
        h = np.dot(o, np.tanh(C))
        self.h.append(h)

        return h

    def forwardPass(self, x):
        self.h = [] # output at time t
        self.C = [] # state at time t

        self.C_bar = []
        self.i = []
        self.f = []
        self.o = []

        self.I = []
        self.z = []

        self.h.append(np.zeros(numCells)) # initial output is empty
        self.C.append(np.zeros(numCells)) # initial state is empty
        self.C_bar.append(np.zeros(numCells)) # this and the following
        # empty arrays make the indexing follow the indexing in papers
        self.i.append(np.zeros(numCells)) 
        self.f.append(np.zeros(numCells)) 
        self.o.append(np.zeros(numCells)) 
        self.W.append(np.zeros(numCells)) 
        self.I.append(np.zeros(numCells)) 
        self.z.append(np.zeros(numCells)) 

        W_1 = np.concatenate((self.W_c, self.U_c), axis=1)
        W_2 = np.concatenate((self.W_i, self.U_i), axis=1)
        W_3 = np.concatenate((self.W_f, self.U_f), axis=1)
        W_4 = np.concateneate((self.W_o, self.U_o), axis=1)

        self.W = np.concatenate((W_1, W_2, W_3, W_4), axis=0)

        outputs = []
        for i in range(len(x)):
            outputs.append(self.forwardStep(x[i]))
        return outputs

    def backwardStep(self, t, dE_dh_t, dE_dc_tplus1):
        
        dE_do_t = np.dot(dE_dh_t, np.tanh(self.C[t]))

        dE_dc_t = dE_dc_tplus1 + np.dot(np.dot(dE_dh_t, self.o[t]), (np.ones(self.numCells) - np.square(np.tanh(self.C[t]))))



        dE_di_t = np.dot(dE_dc_t, self.C_bar[t])

        dE_dcbar_t = np.dot(dE_ct_t, self.i[t])

        dE_df_t = np.dot(dE_ct_t, self.C[t - 1])

        dE_dc_tminus1 = np.dot(dE_dc_t, self.f[t])


        dE_dzcbar_t = np.dot(dE_dcbar_t, (np.ones(self.numCells) - np.square(np.tanh(self.z[t][0:self.numCells]))))
        dE_dzi_t = np.dot(np.dot(dE_di_t, self.i[t]), (np.ones(self.numCells) - self.i[t]))
        dE_dzf_t = np.dot(np.dot(dE_df_t, self.f[t]), (np.ones(self.numCells) - self.f[t]))
        dE_dzo_t = np.dot(np.dot(dE_do_t, self.o[t]), (np.ones(self.numCells) - self.o[t]))

        dE_dz_t = np.concatenate(dE_dzcbar_t, dE_dzi_t, dE_dzf_t, dE_dzo_t)

        dE_dI_t = np.dot(np.transpose(W), dE_dz_t)


        dE_dh_tminus1 = dE_dI_t[self.size:]

        dE_dW_t = np.dot(dE_dz_t, np.transpose(self.I[t])) # this one is confusing cos it says X_t instead of I_t, but there is no matrix or vector X,
        # and the matrix dimensions are correct if we use I instead

        return (dE_dW_t, dE_dh_tminus1, dE_dc_t)



    # Back propagation through time, returns the error and the gradient for this sequence
    # (should I give this the sequence x1,x2,... so that this method is tied
    # to the sequence?)
    def BPTT(self, y, numTimePeriods):
        E = 0.5 * np.square(self.h - y) # This is the error vector for this sequence
        dE_dh_t = self.h - y # This is the error gradient for this sequence
        dE_dc_t = 0

        dE_dW = 0 
        for i in range(numTimePeriods):
            result = backwardStep(numTimePeriods - i, dE_dh_t, dE_dc_t)
            dE_dW = dE_dW + result[0] # dE_dW_t
            dE_dh_t = result[1]
            dE_dc_t = result[2]

        return (E, dE_dW)

class LSTMNetwork:

    # Structure is a vector specifing the structure of the network - each
    # element represents the number of nodes in that layer
    def __init__(self, structure):
        self.layers = [[x for x in structure]] # this doesnt make sense
        for 
