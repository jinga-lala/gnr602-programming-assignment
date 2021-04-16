import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from model import Perceptron
from data_loader import load_sonar_data
np.random.seed(42)

def main(args):
	if args.data == 'or':
		X_train = np.array([[0,0],[0,1], [1,0],[1,1]])
		y_train = np.array([0,1,1,1]) #OR
		# Y = np.array([0,1,1,0]) #XOR
	elif args.data == 'sonar':
		X,Y = load_sonar_data()
		X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.4, random_state=42)
		args.input_dim = 60

	model = Perceptron(args)
	print(X_train, y_train)
	#Training the model
	print('='*20+'Training'+'='*20)
	for epoch in range(args.epoch):
		t = model.forward(X_train,y_train)
		print(f'Epoch {epoch+1}: No. of misclassifications = {abs(model.loss())}')
		model.backward()

	#Testing the model
	if args.data != 'or':
		print('='*20+'Testing'+'='*20)
		t = model.forward(X_test,y_test)
		print(f'No. of misclassifications = {abs(model.loss())} out of {len(y_test)}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Inputs to model run script')
	parser.add_argument('--input_dim', default=2, type=int, help='input dimension of features')
	parser.add_argument('--epoch', default=20000, type=int, help='number of epochs')
	parser.add_argument('--transform', default=False, type=bool, help='do non linear transformation on input or not')
	parser.add_argument('--learning_rate', default=None, type=float, help='learning rate')
	parser.add_argument('--data', default='or', type=str, help='training and eval data')
	args = parser.parse_args()

	main(args)