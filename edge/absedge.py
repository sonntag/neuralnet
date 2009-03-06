# File: absedge.py
# Copyright 2009 Justin Sonntag. All rights reserved.

import autopath
from neuralnet.lib.abstractmethod import *

class absedge (object):

	__metaclass__ = metaclass
	
	def __init__(self, w, no, nt, dir):
		"""creates a new absedge object with weight w and two connection nodes
		no and nt"""
		
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