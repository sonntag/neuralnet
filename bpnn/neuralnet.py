# File: neuralnet.py
# Copyright 2008 Justin Sonntag. All rights reserved.

"""Module for layered syncronous neural networks

Module includes two neural network classes.  nn is a layered, syncronous
neural network with one input layer, one hidden layer, and one output
layer, but it has no learning ability.  bpnn extends on nn by adding
a backpropogating learning method"""

import math
import neuron
import random

random.seed(0)

# calculate a random long where: a <= rand < b
def rand(a, b):
	return (b - a) * random.random() + a
	
# Make a matrix (we could use NumPy to speed this up)
def makematrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# functions for testing
XOR = [[[0,0], [0]], [[0,1], [1]], [[1,0], [1]], [[1,1], [0]]]
OR = [[[0,0], [0]], [[0,1], [1]], [[1,0], [1]], [[1,1], [1]]]
AND = [[[0,0], [0]], [[0,1], [0]], [[1,0], [0]], [[1,1], [1]]]
IMP = [[[0,0], [1]], [[0,1], [1]], [[1,0], [0]], [[1,1], [1]]]
NOT = [[[0], [1]], [[1], [0]]]

# A || (B && C)
EQ = [[[0,0,0], [0]], [[0,0,1], [0]], [[0,1,0], [0]], [[0,1,1], [1]],
	  [[1,0,0], [1]], [[1,0,1], [1]], [[1,1,0], [1]], [[1,1,1], [1]]]

# A && (B || C)
EQ2 = [[[0,0,0], [0]], [[0,0,1], [0]], [[0,1,0], [0]], [[0,1,1], [0]],
	   [[1,0,0], [0]], [[1,0,1], [1]], [[1,1,0], [1]], [[1,1,1], [1]]]

# A -> (B && C)
EQ3 = [[[0,0,0], [1]], [[0,0,1], [1]], [[0,1,0], [1]], [[0,1,1], [1]],
	   [[1,0,0], [0]], [[1,0,1], [0]], [[1,1,0], [0]], [[1,1,1], [1]]]
	   
class nn:
	"""Basic outline of a feed forward, syncronous, layered neural network
	
	This class is a layered neural network at it's basic level.  It does
	not include any learning functionality, and is meant to be inherited
	into other classes that defines this functionality."""

	def __init__(self, ni, nh, no, name = 'nn'):
		self.name = name
		
		# number of input, hidden, and output neurons
		self.ni = ni + 1
		self.nh = nh
		self.no = no
		
		# create neurons
		self.inputnodes = []
		for x in range(self.ni - 1):
			self.inputnodes.append(neuron.neuron(name = 'i' + str(x)))
		self.hiddennodes =[]
		for x in range(self.nh):
			self.hiddennodes.append(neuron.neuron(name = 'h' + str(x)))
		self.outputnodes = []
		for x in range(self.no):
			self.outputnodes.append(neuron.neuron(name = 'o' + str(x)))
			
		# create bias
		self.inputnodes.append(neuron.bias())
		
		# create connections and initialize weights to 0
		for i in self.hiddennodes:
			for j in self.inputnodes:
				j.set_connection(i, 0)
		for i in self.outputnodes:
			for j in self.hiddennodes:
				j.set_connection(i, 0)
	
	def __str__(self):
		temp = self.name + '\n'
	
		# put input neurons on a single line
		for x in self.inputnodes:
			temp += str(x)
		temp += '\n'
		
		# put hidden neurons on a single line
		for x in self.hiddennodes:
			temp += str(x)
		temp += '\n'
		
		# put output neurons on a single line
		for x in self.outputnodes:
			temp += str(x)
		
		return temp
	
	def update(self, inputs):
		if len(inputs) != self.ni - 1:
			raise ValueError, 'wrong number of inputs'
		
		# put inputs into input nodes
		for i in range(self.ni - 1):
			self.inputnodes[i].append(inputs[i])
		
		# activate input nodes
		for i in self.inputnodes:
			i.activate()
			
		# activate hidden nodes
		for i in self.hiddennodes:
			i.activate()
		
		# activate output nodes and store in temp list
		temp = []
		for i in range(self.no):
			temp.append(self.outputnodes[i].activate())
		
		return temp
	
	def setweights(self, wi, wo):
		for x in range(self.ni):
			for y in range(self.nh):
				self.inputnodes[x].set_connection(self.hiddennodes[y], wi[x][y])
		for x in range(self.nh):
			for y in range(self.no):
				self.hiddennodes[x].set_connection(self.outputnodes[y], wo[x][y])

class bpnn(nn):
	"""Network that extends nn to add back propogation
	
	 This class extends the functionality of the nn class to include a
	 back propogating learning function it also includes a method to
	 return an nn object without the learning functionality."""

	def __init__(self, ni, nh, no, name = 'bpnn'):
		nn.__init__(self, ni, nh, no, name)
		
		#self.file = file(name + '.txt', 'w')

		# lists to store the activations
		self.ai = [0.0] * self.ni
		self.ah = [0.0] * self.nh
		self.ao = [0.0] * self.no
		
		# make matrices to store copies of the weights for training
		self.wi = makematrix(self.ni, self.nh, fill = 1.0)
		self.wo = makematrix(self.nh, self.no, fill = 1.0)
		
		# make matrices to store the change in weights for momentum
		self.ci = makematrix(self.ni, self.nh)
		self.co = makematrix(self.nh, self.no)
			
		# create connections and set weights to 1.0
		for i in self.hiddennodes:
			for j in self.inputnodes:
				j.set_connection(i, rand(-2.0, 2.0))
		for i in self.outputnodes:
			for j in self.hiddennodes:
				j.set_connection(i, rand(-2.0, 2.0))
	
	def __nn__(self):
		"returns a neural network without the learning functionality"
		temp = nn(self.ni - 1, self.nh, self.no)
		temp.setweights(self.wi, self.wo)
		return temp
	
	def update(self, inputs):
		if len(inputs) != self.ni - 1:
			raise ValueError, 'wrong number of inputs'
		
		# put inputs into input nodes
		for i in range(self.ni - 1):
			self.inputnodes[i].append(inputs[i])
			
		# activate input nodes and store output into ai
		for i in range(self.ni):
			self.ai[i] = self.inputnodes[i].activate()
		
		# activate hidden nodes and store output into ah
		for i in range(self.nh):
			self.ah[i] = self.hiddennodes[i].activate()
		
		# activate output nodes and store output into ao
		for i in range(self.no):
			self.ao[i] = self.outputnodes[i].activate()
		
		return self.ao
			
	def backpropagate(self, targets, N, M):
		if len(targets) != self.no:
			raise ValueError, 'wrong number of target values'
			
		# get weights from nodes
		for x in range(self.ni):
			for y in range(self.nh):
				self.wi[x][y] = self.inputnodes[x].connections[self.hiddennodes[y]]
		for x in range(self.nh):
			for y in range(self.no):
				self.wo[x][y] = self.hiddennodes[x].connections[self.outputnodes[y]]
				
		# calculate error terms for output
		output_deltas = [0.0] * self.no
		for k in range(self.no):
			error = targets[k] - self.ao[k]
			output_deltas[k] = neuron.dsigmoid(self.ao[k]) * error
		
		# calculate error terms for hidden
		hidden_deltas = [0.0] * self.nh
		for j in range(self.nh):
			error = 0.0
			for k in range(self.no):
				error += output_deltas[k] * self.wo[j][k]
			hidden_deltas[j] = neuron.dsigmoid(self.ah[j]) * error
		
		# update output weights
		for j in range(self.nh):
			for k in range(self.no):
				change = output_deltas[k] * self.ah[j]
				self.wo[j][k] += N * change + M * self.co[j][k]
				self.co[j][k] = change
				#print N * change, M * self.co[j][k]
		
		# update input weights
		for i in range(self.ni):
			for j in range(self.nh):
				change = hidden_deltas[j] * self.ai[i]
				self.wi[i][j] += N * change + M * self.ci[i][j]
				self.ci[i][j] = change
				
		# set new weights to nodes
		self.setweights(self.wi, self.wo)
				
		# calculate error
		error = 0.0
		for k in range(len(targets)):
			error += 0.5 * (targets[k] - self.ao[k]) ** 2
		return error
	
	def test(self, patterns):
		for p in patterns:
			print p[0], '->', self.update(p[0])
	
	def train(self, patterns, iterations = 1000, N = 0.5, M = 0.1):
		# N: learning rate
		# M: momentum factor
		for i in xrange(iterations):
			error = 0.0
			for p in patterns:
				inputs = p[0]
				targets = p[1]
				self.update(inputs)
				error += self.backpropagate(targets, N, M)
			if i % 100 == 0:
				print 'error %-14f' % error
			#self.file.write(str(error) + "\n")
	
def demo():
	# Teach network XOR function
	#pat = [
	#	[[0,0], [0]],
	#	[[0,1], [1]],
	#	[[1,0], [1]],
	#	[[1,1], [0]]
	#]
	
	# create a network with two input, two hidden, and one output nodes
	n = bpnn(2, 2, 1)
	# train it with some patterns
	n.train(XOR)
	# test it
	n.test(XOR)

if __name__ == '__main__':
	demo()