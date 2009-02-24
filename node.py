# File: node.py
# Copyright 2009 Justin Sonntag. All rights reserved.

# Types of neurons:
#	pos: positive neuron with activation between 0 and 1
#	neg: negative neuron with activation between 0 and -1
#	bias: bias neuron with activation equal to 1

class node (object):

	def __init__(self, a = 0.0, l = 0.0, h = 1.0):
		self.activation = a
		self.low = l
		self.high = h