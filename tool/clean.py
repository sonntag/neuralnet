# File: clean.py
# Copyright 2009 Justin Sonntag. All rights reserved

"""Utility file created to remove all files ending in .pyc from
the neuralnet package."""

import autopath

def main():
	
	from os.path import join, walk
	from os import remove
	
	def del_walker(arg, dirname, fnames):
		for fn in fnames:
			if fn.endswith(".pyc"):
				remove(join(dirname, fn))
	
	walk(autopath.nndir, del_walker, None)

if __name__ == "__main__":
	main()
