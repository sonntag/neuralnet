neuralnet is a package written in python enabling the fast development of
neural networks.  Using highly modular objects, one is able to piece together
a neural network and test it's effectiveness.

neuralnet uses a slightly different architecture for creating a neural network
than most people.  The two main classes that all neural networks are
constructed from are the node class and the edge class.  Just like in any
graph, nodes (which hold activations) are connected by weighted edges, and
signals moving between the nodes cause the neural network to compute.

This structure also allows building neuron structures by using a node and a
list of edge objcets.


All code used for references was code put out on the public domain

References:
	bpnn.py by Neil Schemenauer
	autopath.py by the makers of pypy
	Recipe 266468 by Ivo Timmermans
		http://code.activestate.com/recipes/266468/
