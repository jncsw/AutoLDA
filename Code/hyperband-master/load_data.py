"""
load your data, which should be a dictionary with
x_train, y_train, x_test, y_test Numpy arrays
defs files import data from here (from load_data import data)
"""

# this particular example loads data from a pickle file

#import cPickle as pickle
# import pickle

# data_file = 'data/classification.pkl'

print ("loading...")

# with open( data_file, 'rb' ) as f:
# 	data = pickle.load( f )

"""
data is a dict containing numpy arrays: 
{ 'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test }
"""

"-----------------------------------------20200519  Tian Liu for loading Churn ANN data"
#importing the libraries
import numpy as np
import pandas as pd

#importing the dataset
dataset = pd.read_csv('data/Churn_Modelling.csv')
X=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

#encoding catrgorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#label encoding the gender, no need for converting to dummy variable. Because after removing a column to avoid the dummy variable trap, the results would be the same 
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
X=X[:,1:] #avoid dummy vairable trap

#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#feature scaling to ease the computation, COMPULSORY
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.fit_transform(x_test)

#build the dict
data={'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
