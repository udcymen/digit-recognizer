import numpy
from scipy.special import expit
import csv
import matplotlib.pyplot


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        self.wih = numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.who = numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5

        self.activation_function = lambda x: expit(x)

    def train(self, inputs_list, outputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(outputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,  hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.learning_rate * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                                   numpy.transpose(hidden_outputs))
        self.wih += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                                   numpy.transpose(inputs))
        pass

    def query(self, inputs):
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        pass


input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
epochs = 10
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open('train.csv')
training_data_list = training_data_file.readlines()
training_data_file.close()

for e in range(epochs):
    for record in training_data_list[1:]:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

test_data_file = open('test.csv')
test_data_list = test_data_file.readlines()
test_data_file.close()

submission = open('submission.csv', 'w')
submission.writelines(['ImageId,Label\n'])
index = 1
for record in test_data_list[1:]:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values))/255.0 * 0.99 + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    out = str(index) + "," + str(label) + '\n'
    submission.writelines(out)
    index += 1



"""
Show train data
"""
# all_values = training_data_list[1].split(',')
# image_array = numpy.asfarray(all_values[1:]).reshape(28, 28)
# matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
# matplotlib.pyplot.show()
