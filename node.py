# File: node.py
# Copyright 2009 Justin Sonntag. All rights reserved.

import autopath
import numpy as np
from neuralnet.lib.abstractmethod import *

class node (object):
	"""
	node class to act as a neuron in a neural net.
	
	node class has an update method to update the node's activation level
	and a sigmoid function which is applied to the incoming sum before
	the update.
	"""
	
	__metaclass__ = metaclass
	
	def __init__(self, a, l, h):
		self.activation = a
		self.low = l
		self.high = h
	
	update = abstractmethod()
	activate = abstractmethod()
	
	# operator overloads
	
	def __mul__(self, x):
		return self.activation * x.weight
	
	def __str__(self):
		return str(self.activation)

class pos_node (node):

	def __init__(self, a = 0.0, l = 0.0, h = np.inf):
		"""Creates a new node object with activation a, low value 0,
		and high value np.inf"""
		
		super(self.__class__, self).__init__(a, l, h)
	
	def update(self, s):
		"""updates the activation of the node based on an inputted sum s"""
		
		temp = self.activate(s)
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
	activate = staticmethod(sigmoid)
	
