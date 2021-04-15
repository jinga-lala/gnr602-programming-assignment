import numpy as np

class Perceptron(object):
	"""Perceptron"""
	"""Implement a single layer perceptron classifier (input layer + output layer without any
	hidden layer) with a polynomial function for nonlinear transformation of the input. Compare
	this result with the result when no nonlinear transformation of the input is done."""
	def __init__(self, arg):
		super(Perceptron, self).__init__()
		self.arg = arg
		self.weights = np.random.randn(self.arg.input_dim+1).reshape(-1,1) #(d+1,1)

		self.transformation = lambda x: x**2 if self.arg.transform else lambda x: x
		self.learning_rate = args.learning_rate if self.arg.learning_rate is not None else 0.1
		# self.cutoff = args.cutoff if args.cutoff is not None else 0.5 

	def forward(self, x,y):
		self.x = np.concatenate((self.transformation(x), np.ones((x.shape[0],1))), axis=-1) #(?,d+1)
		out = self.x@self.weights #(?,1)
		self.t = (out >= 0) + 0 #(?,1)
		# import pdb;pdb.set_trace()
		self.y = y.reshape(-1,1)
		return self.t

	def backward(self):
		# import pdb;pdb.set_trace()
		self.weights += np.sum(self.learning_rate*(self.y-self.t)*self.x, axis=0).reshape(-1,1)

	def loss(self):
		"""no of incorrect samples"""
		return np.sum(self.y-self.t) 

