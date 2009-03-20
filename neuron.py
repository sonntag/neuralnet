# File: neuron.py
# Copyright 2009 Justin Sonntag. All rights reserved.

import autopath
import numpy as np
from neuralnet.lib.abstractmethod import *

class neuron (object):
	
	__metaclass__ = metaclass
	
	def __init__(self, node, edgelst):
		self.node = node
		self.edgelst = {}
		for e in edgelst:
			if e.no == self.node:
				self.edgelst[e] = e.nt
			else:
				self.edgelst[e] = e.no

	learn = abstractmethod()
	update = abstractmethod()

