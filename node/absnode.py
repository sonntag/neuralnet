# File: absnode.py
# Copyright 2009 Justin Sonntag. All rights reserved.

import autopath
from neuralnet.lib.abstractmethod import *

class absnode (object):
	
	__metaclass__ = metaclass
	
	def __init__(self, a, l, h):
		self.activation = a
		self.low = l
		self.high = h
	
	update = abstractmethod()
	sigmoid = abstractmethod()
	
	def __mul__(self, x):
		return self.activation * x.weight
	
	def __str__(self):
		return str(self.activation)