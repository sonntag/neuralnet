# File: network.py
# Copyright 2009 Justin Sonntag. All rights reserved.

# Used bpnn.py by Neil Schemenauer as a reference

# Written in Python.  See http://www.python.org/
# Requirements:
#		- Python 2.4.x or 2.5.x
#		- NumPy

import numpy as np

HIGH = 1  # high value for a neuron activation
LOW = 0  # low value for a neuron activation

################################################################################

class network:
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
################################################################################
		self.ao = np.ndarray(self.no, int)
		self.ai.fill(LOW)
		self.ah.fill(LOW)
		self.ao.fill(LOW)
		
		# create the weight matrices and set them all to zero
		self.wi = np.ndarray( (self.ni, self.nh) )
		self.wo = np.ndarray( (self.nh, self.no) )
		self.wi.fill(0)
		self.wo.fill(0)
		
		# create hidden weight matrix, a symmetric matrix with all diagonal
		# entries equal to zero
		self.wh = np.ndarray( (self.nh, self.nh) )
################################################################################
		self.wh.fill(0)
		
		