import pandas as pd

def load_sonar_data():
	df = pd.read_csv('data/sonar.all-data', index_col=False, header=None)
	df = df.replace(to_replace='R', value=0)
	df = df.replace(to_replace='M', value=1)
	Y = df[60].to_numpy()
	X = df.to_numpy()[:,:60]
	return X,Y
	


load_sonar_data()
