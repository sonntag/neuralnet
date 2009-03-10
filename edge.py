# File: edge.py
# Copyright 2009 Justin Sonntag. All rights reserved.

import autopath
import numpy as np
from neuralnet.lib.abstractmethod import *

class edge (object):
	"""
	edge class to act as a connection in a neural net.
	
	the edge class is the only thing that is capable to act on a node.
	each edge has a weight and holds the two nodes that it is connected to,
	along with directional information if it is a unidirectional edge.
	"""
	
	__metaclass__ = metaclass
	
	def __init__(self, w, no, nt, dir):
		"""
		creates a new edge object with weight w and two connection nodes
		no and nt
		"""
		
		self.weight = w
		self.nodeone = no
		self.nodetwo = nt
		self.directional = dir
	
	learn = abstractmethod()
	
	# operator overloads
	
	def __mul__(self, x):
		return self.weight * x.activation
	
	def __str__(self):
		return str(self.weight)

class bidirect_edge (edge):
	"""
	edge object that does not have a specified direction
	"""
	
	def __init__(self, w, no, nt):
		"""
		creates a new bidirect_edge object with weight w and two connected
		nodes o and t
		"""
		
		#super(bidirect_edge, self).__init__(w, no, nt, False)
		super(self.__class__, self).__init__(w, no, nt, False)
		
	def learn():
		pass
		
