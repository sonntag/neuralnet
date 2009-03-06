# Modified from Recipe 266468 by Ivo Timmermans, 2004
# http://code.activestate.com/recipes/266468/

class abstractmethod (object):
	pass
			
class metaclass (type):

	def __init__(cls, name, bases, *args, **kwargs):
		super(metaclass, cls).__init__(cls, name, bases, *args, **kwargs)
		
		cls.__new__ = staticmethod(cls.new)
		
		abstractmethods = []
		ancestors = list(cls.__mro__)
		ancestors.reverse()
		for ancestor in ancestors:
			for clsname, clst in ancestor.__dict__.items():
				if isinstance(clst, abstractmethod):
					abstractmethods.append(clsname)
				else:
					if clsname in abstractmethods:
						abstractmethods.remove(clsname)
		
		abstractmethods.sort()
		setattr(cls, '__abstractmethods__', abstractmethods)
	
	def new(self, cls, *args, **kwargs):
		if len(cls.__abstractmethods__):
			raise NotImplementedError('Can\'t instantiate class `' + cls.__name__ + '\';\n' +
										'Abstract methods: ' + ", ".join(cls.__abstractmethods__))
		
		return object.__new__(self)