# File: node.py
# Copyright 2009 Justin Sonntag. All rights reserved.

import numpy as np

class node (object):

	def __init__(self, a = 0.0, l = 0.0, h = np.inf):
		"""Creates a new node object with activation a, low value 0,
		and high value np.inf"""
		
		self.activation = a
		self.low = l
		self.high = h
	
	def update(self, s):
		"""updates the activation of the node based on an inputted sum s"""		
		
		temp = self.sigmoid(s)
		if self.low <= temp <= self.high:
			self.activation = temp
		elif temp < self.low:
			self.activation = self.low
		else:
			self.activation = self.high
	
	def sigmoid(x):
		"""sigmoid function for node activations.  based on input, outputs
		a number between 0 and 1"""
	
		return np.tanh(x)
	sigmoid = staticmethod(sigmoid)
	
	# operator overloads	
	def __add__(self, x):
		if isinstance(x, type(self)):
			return self.activation + x.activation
		return self.activation + x
	
	def __radd__(self, x):
		return self.activation + x
	
	def __sub__(self, x):
		if isinstance(x, type(self)):
			return self.activation - x.activation
		return self.activation - x
	
	def __rsub__(self, x):
		return self.activation - x
	
	def __mul__(self, x):
		if isinstance(x, type(self)):
			return self.activation * x.activation
		return self.activation * x
	
	def __rmul__(self, x):
		return self.activation * x