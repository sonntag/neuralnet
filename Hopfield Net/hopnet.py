# File: hopnet.py
# Copyright 2009 Justin Sonntag. All rights reserved.

import numpy as np


class hopneuron:
	"An implementation of a simple neuron for a hopfield network"
	
	def __init__(self, learnrate, activation = -1, threshold = 0):
		self.learnrate = learnrate
		self.activation = activation
		self.threshold = threshold
	
	def __str__(self):
		return str(self.activation)
	
	def activate(self, energy):
		if energy > self.threshold:
			self.activation = 1
		else self.activation = -1


class hopnet:
	"An implementation of a Hopfield Neural Network"
	
	def __init__(self, size, learnrate):
		self.neurons = np.array([hopneuron(learnrate) for n in range(size)])
		self.weights = np.array([0] * (size ** 2)).reshape(size, size)
		self.size = size
	
	def energy(self):
		sum = 0
		for j in range(self.size):
			for i in range(j - 1):
				sum += self.weights[i,j] * self.neurons[i].activation * self.neurons[j].activation
		sum /= 2.0
		sum += np.sum([n.threshold * n.activation for n in self.neurons])
		return sum
	
	def run(self, initial, time):
		for c, n in enumerate(initial):
			self.neurons[c].activation = n
		for c in range(time):
			
	