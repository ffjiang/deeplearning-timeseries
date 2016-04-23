# deeplearning-timeseries

References:

Simple explanation of LSTMs:
http://colah.github.io/posts/2015-08-Understanding-LSTMs/
http://deeplearning.net/tutorial/lstm.html

An implementation of LSTMs:
http://nicodjimenez.github.io/2014/08/08/lstm.html

A more theoretical textbook, but one which uses a summation over the input
and output elements rather than leaving in their vector form
http://www.cs.toronto.edu/~graves/preprint.pdf

Basis for my implementation of the LSTM algorithm:
http://arunmallya.github.io/writeups/nn/lstm/index.html#/

...

LSTM.py is our own implementation of an LSTM unit, and was superceded by the
Keras implementation that we used, which can be found in LSTMNet.py

Simply running "python LSTMNet.py" will train the model, forecast on a test 
set and plot the forecasted values against the actual valuesl. To change
the model between LSTM, RNN and MLP, uncomment the appropriate line between 159
and 162, which chooses the model that is used.

...

DBN References:

In order to run the DBN, simply include a filename with the data formatted, separated by semi-colons:

EX:

0;1;2;3;2;3
0;2;1;3;2;3
...


To change predictions, run dbn.simulate_visible(data) in order to generate a prediction of hidden unit outcome from
input data. Modify the number of input states, output states, and hidden layers/states in the hidden layer in the main function.
For a DBN with 3 input nodes, two layers with 3 nodes each, and 2 output nodes, the following configuration would be:

input: 3
hidden: [3,3]
output: 3

Introduction to RBM: http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/

Introduction to DBN:
http://www.math.kth.se/matstat/seminarier/reports/M-exjobb15/150612a.pdf

Pretty intuitive description of DBN/Uses nolearn package:
http://www.pyimagesearch.com/2014/09/22/getting-started-deep-learning-python/

Potential implementations of DBNs/RBMs:
http://deeplearning.net/tutorial/DBN.html
https://github.com/dnouri/nolearn/tree/master/nolearn
