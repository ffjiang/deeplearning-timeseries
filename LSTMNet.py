from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, ActivityRegularization
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD
import numpy as np
import pylab as pl
import random

def buildModel(timesteps, data_dim, out_dim, lstmSize):
    model = Sequential()
    model.add(LSTM(lstmSize, return_sequences=True,
                    forget_bias_init='one',
                    activation='linear',
                    init='zero',
                    inner_init='zero',
                    W_regularizer=l2(0.01), U_regularizer=l2(0.01),
                    b_regularizer=l2(0.01),
                    input_shape=(timesteps, data_dim)))
    model.add(ActivityRegularization(l2=0.1))
    model.add(Dropout(0.2))
    '''
    model.add(LSTM(lstmSize, return_sequences=True,
                    forget_bias_init='one',
                    activation='linear',
                    init='zero',
                    inner_init='zero',
                    W_regularizer=l2(0.01), U_regularizer=l2(0.01),
                    b_regularizer=l2(0.01)))
    model.add(ActivityRegularization(l2=0.1))
    model.add(Dropout(0.2))
    '''
    model.add(LSTM(lstmSize, return_sequences=False, activation='linear',
                    init='zero',
                    inner_init='zero',
                    W_regularizer=l2(0.01), U_regularizer=l2(0.01),
                    b_regularizer=l2(0.01))) # Only returns last output
    model.add(ActivityRegularization(l2=0.1))
    model.add(Dense(out_dim, activation='linear', init='zero'))#, W_regularizer=l2(0.01), b_regularizer=l2(0.01))) # Linear regression layer

    opt = SGD(lr=7, decay = 0.0003)
    model.compile(loss='mean_absolute_error', optimizer=opt)

    return model

def buildSimpleModel(timesteps, data_dim, out_dim, lstmSize):
    print('Build model...')
    model = Sequential()
    model.add(LSTM(out_dim, activation='sigmoid', return_sequences=False, input_shape=(timesteps, data_dim)))

    model.compile(loss='mean_absolute_error', optimizer='rmsprop')

    return model

def buildMLP(data_dim, out_dim, mlpSize):
    print('Build MLP...')
    model = Sequential()
    model.add(Dense(mlpSize, activation='relu', input_dim=data_dim))#, W_regularizer=l2(0.01), b_regularizer=l2(0.01))) # Linear regression layer
    model.add(Dense(mlpSize, activation='relu'))
    model.add(Dense(out_dim, activation='relu'))

    model.compile(loss='mean_absolute_error', optimizer='rmsprop')

    return model

def buildRNN(timesteps, data_dim, out_dim):
    print('Build RNN...')
    model = Sequential()
    model.add(SimpleRNN(out_dim, input_shape=(timesteps, data_dim)))

    model.compile(loss='mean_absolute_error', optimizer='rmsprop')


    return model

def train(model, trainingData, numEpochs, sequenceLength):
    # Cut the time series data into semi-redundant sequences of 
    # sequenceLength examples
    step = 50
    trainingSequences = []
    trainingTargets = []

    for i in range(0, trainingData.shape[0] - sequenceLength, step):
        trainingSequences.append(trainingData[i: i + sequenceLength])
        trainingTargets.append(trainingData[i + sequenceLength, 2:]) # Ignore the month and time columns

    trainingSequences = np.array(trainingSequences)
    trainingTargets = np.array(trainingTargets)


    print('Training...')
    history = model.fit(trainingSequences, trainingTargets, batch_size=1, nb_epoch=numEpochs, 
                    shuffle=True, validation_split=0.1, verbose=1)






def readData(filename):
    lines = []  
    with open(filename) as f:
        lines = f.readlines()

    num_attributes = len(lines[0].split(";")) - 1
    del lines[0]

    i = 0
    examples = []
    for line in lines:
        if i < 100000 and '?' not in line:

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
    max_ex -= min_ex
    max_ex += 1
    training_data /= max_ex

    return (training_data, max_ex, min_ex, original_data)



def main():

    sequenceLength = 2000
    numEpochs = 10

    data = readData('household_power_consumption.txt')
    corpusData = data[0]
    max_ex = data[1]
    min_ex = data[2]
    trainingData = corpusData[:-10000]
    forecastData = corpusData[-10000:]

    model = buildSimpleModel(sequenceLength, corpusData.shape[1], 
                        corpusData.shape[1] - 2, 32)
    #model = buildRNN(sequenceLength, corpusData.shape[1], corpusData.shape[1] - 2)
    #model = buildMLP(corpusData.shape[1], corpusData.shape[1] - 2, 16)

    train(model, trainingData, numEpochs, sequenceLength)

    
    forecastInput = []
    forecastInput.append(forecastData[:sequenceLength])
    forecastInput = np.array(forecastInput)
    forecasts = []
    for i in range(len(forecastData) - sequenceLength):
        forecast = model.predict(forecastInput)
        forecasts.append(forecast[0])

        # Remove the oldest example from the input data
        prevInput = forecastInput[0]
        prevInput = prevInput[1:]

        # Add the month and time field back in
        forecastWithTime = [np.concatenate((forecastData[sequenceLength + i, 0:2], forecast[0]))]
        forecastWithTime = np.array(forecastWithTime)

        # Add the newly generated prediction to the input data
        #forecastInput[0] = np.concatenate((prevInput, forecastWithTime), axis=0)
        forecastInput = []
        forecastInput.append(forecastData[i + 1: sequenceLength + i + 1])
        forecastInput = np.array(forecastInput)



    # Plot predictions against labels
    forecasts = np.array(forecasts)
    forecasts *= max_ex[2:]
    forecasts += min_ex[2:]
    forecastData *= max_ex
    forecastData += min_ex
    times = [i for i in range(forecasts.shape[0])]
    times = np.array(times)
    pl.plot(times, forecasts[:,0], 'r')
    pl.plot(times, forecastData[forecastData.shape[0] - forecasts.shape[0]:,2], 'b')
    pl.show()


if __name__ == "__main__": main()
