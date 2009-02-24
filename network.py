# File: network.py
# Copyright 2009 Justin Sonntag. All rights reserved.

# Used bpnn.py by Neil Schemenauer as a reference

# Written in Python.  See http://www.python.org/
# Requirements:
#		- Python 2.4.x or 2.5.x
#		- NumPy

"""network.py is a module created for developing neural networks for research
purposes in python.  It defines a network class that creates a syrcronous,
feedback enabled neural network used to test training methods."""


import numpy as np
from numpy import random

random.seed(0)

HIGH = 1  # high value for a neuron activation
LOW = 0  # low value for a neuron activation
THREASHOLD = 0.5  # threashold value for neuron activation

class network (object):
	"""Neural network with an input and output layer, and a hidden network"""
	
	def __init__(self, ni, nh, no):
		"""Creates a new network with ni input nodes, nh hidden nodes,
		and no output nodes"""
		
		self.ni = ni + 1  # +1 for the bias node (to be replaced in the future)
		self.nh = nh
		self.no = no
		
		# create the neuron activations and set them to the low value
		self.ai = np.ndarray(self.ni, int)
		self.ah = np.ndarray(self.nh, int)
		self.ao = np.ndarray(self.no, int)
		self.ai.fill(LOW)
		self.ah.fill(LOW)
		self.ao.fill(LOW)
		
		# create the weight matrices and set them to random values
		self.wi = random.rand(self.nh, self.ni)
		self.wo = random.rand(self.no, self.nh)
		
		# create hidden weight matrix, a symmetric matrix with all diagonal
		# entries equal to zero
		self.wh = np.ndarray( (self.nh, self.nh) )
		self.wh.fill(0)
		
		for i in xrange(self.nh):
			for j in xrange(i+1, self.nh):
				self.wh[i,j] = self.wh[j,i] = random.rand()
		
	def update(self, inputs):
		"""Updates the value of the neuron activations based on the given
		input"""
		
		if len(inputs) != self.ni - 1:
			raise ValueError, "wrong number of inputs"
		
		# set input node activations
		# (assumes that inputs are either LOW or HIGH)
		for c, v in enumerate(inputs):
			self.ai[c] = v
			
		# set hidden node activations
		self.ah = self.sigmoid(np.dot(self.wi, self.ai)
						+ np.dot(self.wh, self.ah))
		
		# set output node activations
		self.ao = self.sigmoid(np.dot(self.wo, self.ah))
		
		return self.ao
	
	def sigmoid(self, x):
		"""sigmoid function for node activations.  based on input, outputs
		a number between 0 and 1"""
		
		return np.tanh(x)
