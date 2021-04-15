import numpy as np
import argparse
from model import Perceptron


def main(args):
	X = np.array([[0,0],[0,1], [1,0],[1,1]])
	Y = np.array([0,1,1,1])

	model = Perceptron(args)
	for epoch in range(40):
		t = model.forward(X,Y)
		print(f'Epoch {epoch+1}: {model.loss()}')
		model.backward()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Inputs to data loading script')
	parser.add_argument('--input_dim', default=2, type=int, help='input dimension of features')
	parser.add_argument('--transform', default=False, type=bool, help='do non linear transformation on input or not')
	parser.add_argument('--learning_rate', default=None, type=float, help='learning rate')
	args = parser.parse_args()

	main(args)