# File: autopath.py
# Copyright 2009 Justin Sonntag. All rights reserved.

def __dirinfo(part):
	
	import sys, os
	
	head = this_dir = os.path.realpath(os.path.dirname(__file__))
	
	while head:
		partdir = head
		head, tail = os.path.split(head)
		if tail == part:
			break
	else:
		raise EnvironmentError("Invalid source tree! Cannot find the " +
			"parent directory %r of the path %r" % (partdir, this_dir))
	
	#nn_root = os.path.join(head, "")
	
	try:
		sys.path.remove(head)
	except ValueError:
		pass
	sys.path.insert(0, head)
	
	return partdir, this_dir

def __clone():
	
	from os.path import join, walk
	if not this_dir.endswith(join("neuralnet", "tool")):
		raise EnvironmentError("can only clone master version "
								"'%s'" % join(nndir, "tool", _myname))
	
	def sync_walker(arg, dirname, fnames):
		if _myname in fnames:
			fn = join(dirname, _myname)
			f = open(fn, 'rwb+')
			try:
				if f.read() == arg:
					print "checkok", fn
				else:
					print "syncing", fn
					f = open(fn, 'w')
					f.write(arg)
			finally:
				f.close()
	s = open(join(nndir, "tool", _myname), 'rb').read()
	walk(nndir, sync_walker, s)

_myname = "autopath.py"

nndir, this_dir = __dirinfo("neuralnet")

if __name__ == "__main__":
	__clone()
