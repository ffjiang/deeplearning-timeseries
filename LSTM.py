import numpy as np
import math
import random
import pylab as pl

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LSTMCell: 

    # Size is the dimensionality of the input vector
    def __init__(self, inputSize, numCells):
        self.inputSize = inputSize
        self.numCells = numCells

        # Randomly initialise the weight matrix
        self.W = np.random.random((4 * numCells, inputSize + numCells)) * 2 \
                        - np.ones((4 * numCells, inputSize + numCells))

        self.h = []
        self.C = []

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
        C = np.multiply(f, self.C[-1]) + np.multiply(i, C_bar)
        self.C.append(C)

        # Compute the new output
        h = np.multiply(o, np.tanh(C))
        self.h.append(h)

        return h

    def forwardPass(self, x):
        self.h = []
        self.C = []

        self.C_bar = []
        self.i = []
        self.f = []
        self.o = []

        self.I = []
        self.z = []

        numCells = self.numCells
        self.h.append(np.zeros(numCells)) # initial output is empty
        self.C.append(np.zeros(numCells)) # initial state is empty
        self.C_bar.append(np.zeros(numCells)) # this and the following
        # empty arrays make the indexing follow the indexing in papers
        self.i.append(np.zeros(numCells)) 
        self.f.append(np.zeros(numCells)) 
        self.o.append(np.zeros(numCells)) 
        self.I.append(np.zeros(numCells)) 
        self.z.append(np.zeros(numCells)) 


        outputs = [self.forwardStep(x_t) for x_t in x]

        return outputs

    def backwardStep(self, t, dE_dh_t, dE_dc_tplus1):
        
        dE_do_t = np.multiply(dE_dh_t, np.tanh(self.C[t]))

        dE_dc_t = dE_dc_tplus1 + np.multiply(np.multiply(dE_dh_t, self.o[t]), (np.ones(self.numCells) - np.square(np.tanh(self.C[t]))))



        dE_di_t = np.multiply(dE_dc_t, self.C_bar[t])

        dE_dcbar_t = np.multiply(dE_dc_t, self.i[t])

        dE_df_t = np.multiply(dE_dc_t, self.C[t - 1])

        dE_dc_tminus1 = np.multiply(dE_dc_t, self.f[t])


        dE_dzcbar_t = np.multiply(dE_dcbar_t, (np.ones(self.numCells) - np.square(np.tanh(self.z[t][0:self.numCells]))))
        dE_dzi_t = np.multiply(np.multiply(dE_di_t, self.i[t]), (np.ones(self.numCells) - self.i[t]))
        dE_dzf_t = np.multiply(np.multiply(dE_df_t, self.f[t]), (np.ones(self.numCells) - self.f[t]))
        dE_dzo_t = np.multiply(np.multiply(dE_do_t, self.o[t]), (np.ones(self.numCells) - self.o[t]))

        dE_dz_t = np.concatenate((dE_dzcbar_t, dE_dzi_t, dE_dzf_t, dE_dzo_t))

        dE_dI_t = np.dot(np.transpose(self.W), dE_dz_t)


        dE_dh_tminus1 = dE_dI_t[self.inputSize:]

        dE_dz_t.shape = (len(dE_dz_t), 1)
        self.I[t].shape = (len(self.I[t]), 1)
        dE_dW_t = np.dot(dE_dz_t, np.transpose(self.I[t])) # this one is confusing cos it says X_t instead of I_t, but there is no matrix or vector X,
        # and the matrix dimensions are correct if we use I instead

        return (dE_dW_t, dE_dh_tminus1, dE_dc_tminus1)



    # Back propagation through time, returns the error and the gradient for this sequence
    # (should I give this the sequence x1,x2,... so that this method is tied
    # to the sequence?)
    def BPTT(self, y):
        numTimePeriods = len(y)
        dE_dW = 0 
        dE_dh_t = 0
        dE_dc_t = 0
        E = 0.0
        discount = 1.0
        for i in range(numTimePeriods):
            index = numTimePeriods - i
            #E = E + 0.5 * np.sum(np.square(self.h[index] - y[index - 1])) # This is the error/loss vector for this sequence
            E = E + discount * np.sum(np.absolute(self.h[index] - y[index - 1])) # This is the error/loss vector for this sequence
            # The gradient is just 1 or -1, depending on whether h is
            # less than or greater than y
            lessThan = np.less(self.h[index], y[index - 1])
            greaterThan = np.greater(self.h[index], y[index - 1])
            dE_dh_t -= discount * lessThan
            dE_dh_t += discount * greaterThan
                
            #dE_dh_t += self.h[index] - y[index - 1] # This is the error gradient for this sequence

            result = self.backwardStep(index, dE_dh_t, dE_dc_t)
            dE_dW = dE_dW + result[0] # dE_dW_t
            dE_dh_t = result[1]
            dE_dc_t = result[2]

            discount *= 0.99

        return (E / (numTimePeriods), dE_dW)


    # should the actul data be used as the next input, or should its own input
    # be used as the next input?
    def train(self, trainingData, numEpochs, learningRate, sequenceLength):
        adaptiveLearningRate = learningRate
        for epoch in range(numEpochs):
            trainingSequences = sequenceProducer(trainingData, sequenceLength)
            epochError = 0.0
            counter = 0
            for sequence in trainingSequences:
                counter += 1
                self.forwardPass(sequence[:-1])
                result = self.BPTT(sequence[1:,3:])
                E = result[0]
                dE_dW = result[1]

                # Annealing
                adaptiveLearningRate = learningRate / (1 + (epoch/10))
                
                self.W = self.W - adaptiveLearningRate * dE_dW

                epochError += E

            print('Epoch ' + str(epoch) + ' error: ' + str(epochError / counter))


    # needs a parameter about how far to forecast, and needs to use its own
    # results as inputs to the next thing, to keep forecasting
    def forecast(self, forecastingData):
        self.forwardPass(forecastingData)
        return np.transpose(np.transpose(self.h[-1]))

    def forecastKSteps(self, forecastingData, timeData, k):
        self.forwardPass(forecastingData)

        for i in range(k - 1):
            lastForecast = self.h[-1]
            nextInput = np.concatenate(([1], timeData[i], self.h[-1]), axis=1)
            self.forwardStep(nextInput)

        return np.transpose(np.transpose(self.h[-k:]))



    # needs fixing
    def test(self, testingData, sequenceLength):
        avgError = 0.0
        testingSequences = sequenceProducer(testingData, sequenceLength)
        counter = 0
        for sequence in testingSequences:
            counter += 1
            self.forwardPass(sequence[:-1])
            E = 0.0
            for j in range(sequenceLength - 1):
                index = sequenceLength - j - 1
                E = E + 0.5 * np.sum(np.square(self.h[index] - sequence[index, 3:])) # This is the error vector for this sequence
            E = E / sequenceLength
            avgError = avgError + E
        avgError = avgError / counter

        return avgError

            

def readData(filename):
    lines = []  
    with open(filename) as f:
        lines = f.readlines()

    num_attributes = len(lines[0].split(";")) - 1
    del lines[0]

    i = 0
    examples = []
    for line in lines:
        if i < 10000 and '?' not in line:

            tokens = line.split(';')
            date_str = tokens[0]
            month = float(date_str.split('/')[1])
            time_str = tokens[1]
            hours = float(time_str[:2])
            minutes = float(time_str[3]) / 60.0
            time = hours + minutes

            global_active_power = float(tokens[2])
            global_reactive_power = float(tokens[3])
            voltage = float(tokens[4])
            global_intensity = float(tokens[5])
            sub_metering_1 = float(tokens[6])
            sub_metering_2 = float(tokens[7])
            sub_metering_3 = float(tokens[8])

            example = [month, time, global_active_power, global_reactive_power, voltage, global_intensity, sub_metering_1, sub_metering_2, sub_metering_3]
            examples.append(example)
            i += 1

    training_data = np.array(examples)
    min_ex = np.amin(training_data, axis=0)
    max_ex = np.amax(training_data, axis=0)

    original_data = np.copy(training_data)
    training_data -= min_ex
    training_data /= max_ex

    return (training_data, max_ex, min_ex, original_data)

'''
class LSTMNetwork:

    # Structure is a vector specifing the structure of the network - each
    # element represents the number of nodes in that layer
    def __init__(self, structure):
        self.layers = [[x for x in structure]] # this doesnt make sense
'''

def sequenceProducer(trainingData, sequenceLength):
    indices = [i for i in range(0, trainingData.shape[0] - sequenceLength + 1, sequenceLength)]
    random.shuffle(indices)
    for index in indices:
        yield trainingData[index:index + sequenceLength]

def forecastSequenceProducer(trainingData, sequenceLength):
    for i in range(trainingData.shape[0] - sequenceLength + 1):
        yield trainingData[i:i + sequenceLength]

def main():

    sequenceLength = 100
    #xSet = np.array([[[1,2,3], [1,3,5], [9, 9, 9]], [[1, 2, 3], [9, 9, 9]]])
    #ySet = np.array([[[0.3, 0.5], [0.3,0.5], [0.3,0.5]], [[0.3, 0.5], [0.3,0.5]]])

    data = readData('household_power_consumption.txt')
    corpusData = data[0]
    corpusData = np.concatenate((np.ones((corpusData.shape[0], 1)), corpusData), axis=1)
    
    lstm = LSTMCell(corpusData.shape[1], corpusData.shape[1] - 3)
    trainingData = corpusData[:-1000]
    lstm.train(trainingData, 30, 0.05, sequenceLength) # data, numEpochs, learningRate, sequenceLength

    testingData = corpusData[-500:]
    #print("Test error: " + str(lstm.test(testingData, sequenceLength)))

    max_ex = data[1]
    min_ex = data[2]

    originalData = data[3]
    forecastData = corpusData[-500:]
    forecastSequences = forecastSequenceProducer(forecastData, sequenceLength)
    forecastError = 0.0
    countForecasts = 0
    labels = []

    forecasts = []
    for sequence in forecastSequences: 
        countForecasts += 1
        forecast = lstm.forecast(sequence[:-1])
        forecast *= max_ex[2:]
        forecast += min_ex[2:]
        label = sequence[-1,3:] * max_ex[2:]
        label += min_ex[2:]

        forecasts.append(forecast)
        labels.append(label)

        print(str(forecast))
        print(str(label))
        print('Error: ' + str(np.absolute(forecast - label)))
        print('----------------')
        forecastError += np.absolute(forecast - label)


    print ('Average forecast error: ' + str(forecastError / countForecasts))
    
    forecasts = np.array(forecasts)
    labels = np.array(labels)
    times = [i for i in range(forecasts.shape[0])]
    pl.plot(times, forecasts[:,0], 'r')
    pl.plot(times, labels[:,0], 'b')
    pl.show()


    '''
    forecasts = lstm.forecastKSteps(forecastData[:500], forecastData[500:,1:3], 500)
    print(forecasts)
    forecasts *= max_ex[2:]
    forecasts += max_ex[2:]
    labels = forecastData[:,3:]
    labels *= max_ex[2:]
    labels += min_ex[2:]


    times = [i for i in range(labels.shape[0])]
    pl.plot(times, np.concatenate((np.ones(500),forecasts[:,0])), 'r')
    pl.plot(times, labels[:,0], 'b')
    pl.show()
    #print('Error: ' + str(lstm.test(xTest, yTest) * max_ex[1] + min_ex[1]))
    '''

if __name__ == "__main__": main()

