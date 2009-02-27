# File: node.py
# Copyright 2009 Justin Sonntag. All rights reserved.

# Types of neurons:
#	pos: positive neuron with activation between 0 and 1
#	neg: negative neuron with activation between 0 and -1
#	bias: bias neuron with activation equal to 1

import numpy as np

class node (object):

	def __init__(self, a = 0.0, l = 0.0, h = 1.0):
		"""Creates a new node object with activation a, low value 0,
		and high value 1"""
		
		self.activation = a
		self.low = l
		self.high = h
	
	def update(self, s):
		if self.low <= s <= self.high:
			self.activation = s
		elif s < self.low:
			self.activation = self.low
		else:
			self.activation = self.high
	
	def sigmoid(x):
		"""sigmoid function for node activations.  based on input, outputs
		a number between 0 and 1"""
	
		return np.tanh(2 * x)
	sigmoid = staticmethod(sigmoid)
