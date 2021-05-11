import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from model import Perceptron
from data_loader import load_sonar_data
np.random.seed(0)

def main(args):
	if args.data == 'or':
		X_train = np.array([[0,0],[0,1], [1,0],[1,1]])
		y_train = np.array([0,1,1,1]) #OR
		# Y = np.array([0,1,1,0]) #XOR
	elif args.data == 'sonar':
		X,Y = load_sonar_data()
		X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.4, random_state=42)
		args.input_dim = 60
	elif args.data == 'toy':
		X_train = np.array([[-6],[-3],[-2],[-1],[0],[1],[2],[3],[4],[5],[6]])
		y_train = np.array([0,0,1,1,1,1,1,0,0,0,0])
		X_test = np.array([[-0.1], [0.1], [1.2],[2.3],[-2.2],[-1.8],[1.9],[3.5], [3], [4], [-3], [-4]])
		y_test = np.array([1,1,1,0,0,1,1,0,0,0,0,0])
		args.input_dim = 1


	model = Perceptron(args)
	# print(X_train, y_train)
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
		# print(f'No. of misclassifications = {abs(model.loss())} out of {len(y_test)}')
		print(f'Test accuracy: {100-100.00*abs(model.loss())/ len(y_test)}%')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Inputs to model run script')
	parser.add_argument('--input_dim', default=2, type=int, help='input dimension of features')
	parser.add_argument('--epoch', default=20000, type=int, help='number of epochs')
	parser.add_argument('--transform', default=0, type=int, help='do non linear transformation on input or not')
	parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
	parser.add_argument('--data', default='or', type=str, help='training and eval data')
	args = parser.parse_args()

	main(args)