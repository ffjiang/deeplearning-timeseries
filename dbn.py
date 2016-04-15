from __future__ import print_function
import numpy as np
from rbm import RBM

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

	def simulate(self, data):

		curr_input = data

		for i in range(0, len(self.rbm_layers)):

			rbm_layer = self.rbm_layers[i]
			curr_input = rbm_layer.run_visible(curr_input)

		return curr_input


if __name__ == '__main__':
	training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
	d = DBN(n_ins=6, hidden_layers=[2], n_outs=2, learning_rate=.1, max_epochs=10000)
	d.train(training_data)
  	for i in range(0, len(d.rbm_layers)):
  		print(d.rbm_layers[i].weights)
  	user = np.array([[0,0,0,1,1,0]])
  	print(d.simulate(user))



