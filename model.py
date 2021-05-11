import numpy as np
np.random.seed(0)

class Perceptron(object):
	"""Perceptron"""
	"""Implement a single layer perceptron classifier (input layer + output layer without any
	hidden layer) with a polynomial function for nonlinear transformation of the input. Compare
	this result with the result when no nonlinear transformation of the input is done."""
	def __init__(self, arg):
		super(Perceptron, self).__init__()
		self.arg = arg
		self.weights = np.random.randn(self.arg.input_dim+1).reshape(-1,1) #(d+1,1)
		def square(x):
			return x**2
		# self.transformation = la.mbda x: x**2 if self.arg.transform==True else x
		print(self.arg.transform)
		if self.arg.transform==1:
			self.transformation = square
		else:
			self.transformation = lambda x: x
		self.learning_rate = arg.learning_rate if self.arg.learning_rate is not None else 0.1
		# self.cutoff = args.cutoff if args.cutoff is not None else 0.5 

	def forward(self, x,y):
		# import pdb;pdb.set_trace()
		self.x = np.concatenate((self.transformation(x), np.ones((x.shape[0],1))), axis=-1) #(?,d+1)
		# print(self.x)
		out = self.x@self.weights #(?,1)
		self.t = (out >= 0) + 0 #(?,1)
		self.y = y.reshape(-1,1)
		return self.t

	def backward(self):
		self.weights += np.sum(self.learning_rate*(self.y-self.t)*self.x, axis=0).reshape(-1,1)

	def loss(self):
		"""no of incorrect samples"""
		# print(self.x[(self.y-self.t).reshape(-1)==0])
		# print(self.y.reshape(-1))
		# print(self.t.reshape(-1))
		return np.sum(self.y-self.t) 

