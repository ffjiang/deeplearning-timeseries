import numpy as np
from rbm import RBM
import sys
from sklearn import preprocessing

def main():

    args = sys.argv
    filename = args[1]

    training_data = readData(filename)
    print training_data
    # norm_data = preprocessing.scale(training_data)

    print "Length: %d" % len(training_data)
    print "Other_Length: %d" % len(training_data[0])

    user = np.array([[0,.9,.8,0,0,0,0,0]])
    ans = np.zeros(shape=[1,10])
    other_ans = np.zeros(shape=[1,8])
    d = DBN(n_ins=8, hidden_layers=[10], n_outs=10, learning_rate=.3, max_epochs=5000)
    d.train(training_data)
    num_sims = 1000
    for j in range(num_sims):
        val = np.array(d.simulate_visible(user))
        back_val = np.array(d.simulate_hidden(val))
        other_ans += back_val
        ans += val
    print (ans / num_sims)
    print (other_ans / num_sims)

def readData(filename):
    lines = []  
    with open(filename) as f:
        lines = f.readlines()

    num_attributes = len(lines[0].split(";")) - 1
    del lines[0]

    i = 0
    examples = []
    for line in lines:
        if i < 3000 and '?' not in line:

            tokens = line.split(';')
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

            example = [time, global_active_power, global_reactive_power, voltage, global_intensity, sub_metering_1, sub_metering_2, sub_metering_3]
            examples.append(example)
            i += 1

    training_data = np.array(examples)

    min_ex = np.amin(training_data, axis=0)
    max_ex = np.amax(training_data, axis=0)

    print len(training_data)
    print(len(training_data[0]))
    training_data -= min_ex
    training_data *= max_ex
    print len(training_data)
    print(len(training_data[0]))
    return training_data

def normalize(arr):
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]
    for i in range(len(arr)):
        arr[i] /= sum 

    return arr

class DBN:

    def __init__(self, n_ins, hidden_layers, n_outs, learning_rate, max_epochs):

        self.rbm_layers = []
        self.max_epochs = max_epochs
        # hidden layers of rbm + visible input layer
        self.n_layers = len(hidden_layers) 

        for i in range(0, self.n_layers):

            if i == 0:
                rbm_layer = RBM(n_ins, hidden_layers[i], learning_rate)
            else:
                rbm_layer = RBM(hidden_layers[i - 1], hidden_layers[i], learning_rate)
            
            self.rbm_layers.append(rbm_layer)
        output_layer = RBM(hidden_layers[len(hidden_layers) - 1], n_outs, learning_rate)

        self.rbm_layers.append(output_layer)

    def train(self, data):

        curr_input = data

        for i in range(0, len(self.rbm_layers)):

            rbm_layer = self.rbm_layers[i]
            rbm_layer.train(curr_input, self.max_epochs)

            curr_input = rbm_layer.run_visible(curr_input)


    def simulate_visible(self, data):

        curr_input = data

        for i in range(0, len(self.rbm_layers)):

            rbm_layer = self.rbm_layers[i]
            curr_input = rbm_layer.run_visible(curr_input)

        return curr_input

    def simulate_hidden(self, data):
        curr_input = data

        for i in range(len(self.rbm_layers) - 1, -1, -1):

            rbm_layer = self.rbm_layers[i]
            curr_input = rbm_layer.run_hidden(curr_input)

        return curr_input
if __name__ == '__main__':
    main()
    



