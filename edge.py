# File: edge.py
# Copyright 2009 Justin Sonntag. All rights reserved.

import numpy as np

class edge (object):
	"""edge class to act as a connection in a neural net.
	
	the edge class is the only thing that is capable to act on a node.
	each edge has a weight and holds the two nodes that it is connected to,
	along with directional information if it is a unidirectional edge."""
	
	def __init__(self, w, o, t):
		"""creates a new edge object with weight w and two connected nodes
		o and t"""
		self.weight = w
		self.nodeone = o
		self.nodetwo = t
	
	# operator overloads
	
	def __mul__(self, x):
		return self.weight * x.activation
	
	def __str__(self):
		return str(self.weight)