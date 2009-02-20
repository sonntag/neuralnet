# File: neuron.py
# Copyright 2008 Justin Sonntag. All rights reserved.

import math

def sigmoid(x):
	"sigmoid function for the activation"
	return math.tanh(x)
		
def dsigmoid(y):
	"derivative of the sigmoid function"
	return 1.0 - y * y

class neuron:
	"An implementatoin of a basic unit in a neural network"
	
	idincrementer = 0
	
	def __init__(self, name = "neuron"):
		self.activation = 0.0  # holds the value of the activation
		#self.inputs = []  # holds the values that are inputted to the neuron from other neurons
		self.sum = 0.0
		self.connections = {}  # holds the connections to other neurons and their weights
		self.name = name  # holds the name of this neuron
		self.id = neuron.idincrementer  # holds the id of this neuron
		neuron.idincrementer += 1
		# self.bias = 1  # bias for the neuron
	
	def __str__(self):
		return "(" + self.name + " " + str(self.activation) + ")"
		
	def set_connection(self, n, w):
		"Add or update a connection to another neuron and its weight"
		self.connections[n] = w
	
	def remove_connection(self, n):
		"Remove a connection to a neuron"
		self.connections.pop(n)
	
	def update_activation(self):
		"update the activation of the neuron"
		#self.activation = sigmoid(sum(self.inputs))
		self.activation = sigmoid(self.sum)
		#self.inputs = []
		self.sum = 0.0
		#return self.activation
	
	def activate(self):
		"send the activation to the connected neurons"
		# if activation is 0 update
		if self.activation == 0:
			self.update_activation()

		for x in self.connections.iterkeys():
			x.append(self.activation * self.connections[x])
		temp = self.activation
		self.activation = 0.0
		return temp
	
	def append(self, i):
		"appned a new input to the neurons inputs"
		#self.inputs.append(i)
		self.sum += i

class bias(neuron):
	def __init__(self):
		neuron.__init__(self, name = "bias")
		self.activation = 1.0
	
	def activate(self):
		temp = neuron.activate(self)
		self.activation = 1.0
		return temp
		
	def update_activation(self):
		self.sum = 0.0