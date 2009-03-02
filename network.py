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
from node import node

random.seed(0)

class network (object):
	"""Neural network with an input and output layer, and a hidden network"""
	
	def __init__(self, ni, nh, no):
		"""Creates a new network with ni input nodes, nh hidden nodes,
		and no output nodes"""
		
		self.ni = ni + 1  # +1 for the bias node (to be replaced in the future)
		self.nh = nh
		self.no = no
		
		# create the node objects
		self.ai = np.ndarray(self.ni, object)
		self.ah = np.ndarray(self.nh, object)
		self.ao = np.ndarray(self.no, object)
		for x in xrange(self.ni): self.ai[x] = node()
		for x in xrange(self.nh): self.ah[x] = node()
		for x in xrange(self.no): self.ao[x] = node()
		
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
		for c, v in enumerate(inputs):
			self.ai[c] = v
			
		# set hidden node activations
		for c, n in enumerate(self.ah):
			n.update(np.dot(self.wi[c], self.ai)
						+ np.dot(self.wh[c], self.ah))
		
		# set output node activations
		for c, n in enumerate(self.ao):
			n.update(np.dot(self.wo[c], self.ah))
		
		return self.ao
