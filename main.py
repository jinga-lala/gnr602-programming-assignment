import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from model import Perceptron
from data_loader import load_sonar_data
from PIL import Image
from numpy import asarray
import os

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
	elif args.data == 'image':
		image = Image.open(args.train_image)
		data = asarray(image)
		image2 = Image.fromarray(data)
		size = data.shape
		label = Image.open(args.train_label)
		ldata = asarray(label)
		label2 = Image.fromarray(ldata)
		X = []
		Y = []
		for i in range(size[0]):
			for j in range(size[1]):
				X.append(data[i][j]/255)
				Y.append(ldata[i][j]/255)
		X_train = np.array(X)
		Y = np.array(Y).astype('int')
		# assert (Y[:,0] == Y[:,1]).all() and (Y[:,0] == Y[:,2]).all()
		y_train = Y[:]

		image = Image.open(args.test_image)
		data = asarray(image)
		image2 = Image.fromarray(data)
		size = data.shape
		label = Image.open(args.test_label)
		ldata = asarray(label)
		label2 = Image.fromarray(ldata)
		X = []
		Y = []
		for i in range(size[0]):
			for j in range(size[1]):
				X.append(data[i][j]/255)
				Y.append(ldata[i][j]/255)
		X_test = np.array(X)
		Y = np.array(Y).astype('int')
		shp = ldata.shape
		# assert (Y[:,0] == Y[:,1]).all() and (Y[:,0] == Y[:,2]).all()
		y_test = Y[:]

		# X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
		args.input_dim = 3  # 3 channel input


	model = Perceptron(args)
	# print(X_train, y_train)
	#Training the model
	print('='*20+'Training'+'='*20)
	for epoch in range(args.epoch):
		t = model.forward(X_train,y_train)
		print(f'Epoch {epoch+1}: No. of misclassifications = {abs(model.loss())}')
		model.backward()

	#Testing the model
	if args.data == 'image':
		print('='*20+'Testing'+'='*20)
		t = model.forward(X_test,y_test)
		out = t.reshape(shp)
		im = Image.fromarray(np.uint8(out*255))
		im.save(os.path.join(args.out_image, 'output.png'))
		print(f'Test accuracy: {100-100.00*abs(model.loss())/ len(y_test)}%')

	elif args.data != 'or':
		print('='*20+'Testing'+'='*20)
		t = model.forward(X_test,y_test)
		# print(f'No. of misclassifications = {abs(model.loss())} out of {len(y_test)}')
		print(f'Test accuracy: {100-100.00*abs(model.loss())/ len(y_test)}%')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Inputs to model run script')
	parser.add_argument('--input_dim', default=2, type=int, help='input dimension of features')
	parser.add_argument('--epoch', default=None, type=int, help='number of epochs')
	parser.add_argument('--transform', default=None, type=int, help='do non linear transformation on input or not')
	parser.add_argument('--learning_rate', default=None, type=float, help='learning rate')
	parser.add_argument('--data', default=None, type=str, help='training and eval data')
	parser.add_argument('--train_image', default=None, type=str, help='Training image path')
	parser.add_argument('--train_label', default=None, type=str, help='Label image path')
	parser.add_argument('--test_image', default=None, type=str, help='Test image path')
	parser.add_argument('--test_label', default=None, type=str, help='Test label path')
	parser.add_argument('--out_image', default=None, type=str, help='Predicted image store path')
	args = parser.parse_args()

	import tkinter as tk
	from tkinter import simpledialog

	ROOT = tk.Tk()

	ROOT.withdraw()
	ent1=tk.Entry(ROOT,font=40)
	ent1.grid(row=2,column=2)
	from tkinter.filedialog import askopenfilename
	from tkinter import filedialog


	# def browsefunc():
	# 	filename = askopenfilename(filetypes=(("jpg file", "*.jpg"), ("png file ",'*.png'), ("All files", "*.*"),))
	# 	ent1.insert(END, filename) # add this	
	# 	b1=Button(ROOT,text="DEM",font=40,command=browsefunc)
	# 	b1.grid(row=2,column=4)
	# browsefunc()
	
	# the input dialog
	if args.transform is None:
		args.transform = simpledialog.askstring(title="Transformation",
									  prompt="Do you want non-linear transformation on input? Enter 0 for No, 1 for Yes \n Press OK for default value")
	args.transform = 0 if args.transform == '' else int(args.transform)

	if args.data is None:       
		args.data = simpledialog.askstring(title="Data",
									  prompt="Which dataset do you want to use? Possible choices [or, toy, sonar, image] \n Press OK for default value")
	args.data = 'or' if args.data == '' else args.data
	if args.data == 'image':
		currdir = os.getcwd()
		args.train_image = filedialog.askopenfilename(parent=ROOT, initialdir=currdir, title='Please select train image')
		args.train_label = filedialog.askopenfilename(parent=ROOT, initialdir=currdir, title='Please select train label')
		args.test_image = filedialog.askopenfilename(parent=ROOT, initialdir=currdir, title='Please select test image')
		args.test_label = filedialog.askopenfilename(parent=ROOT, initialdir=currdir, title='Please select test label')
		args.out_image = filedialog.askdirectory(parent=ROOT, initialdir=currdir, title='Please select a directory to store test output')
		print(args)
	if args.learning_rate is None:
		args.learning_rate = simpledialog.askstring(title="Learning rate",
									  prompt="Set the learning rate for this run \n Press OK for default value")
	args.learning_rate = 0.1 if args.learning_rate == '' else float(args.learning_rate)

	if args.epoch is None:
		args.epoch = simpledialog.askstring(title="Epochs",
									  prompt="Set the number of epochs for this run \n Press OK for default value")
	args.epoch  = 100 if args.epoch == '' else int(args.epoch)

	# check it out


	main(args)