# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('Emission_Co2.csv')

data1 = dataset[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

X = data1.iloc[:, :6]



#X['FUELTYPE'] = X['FUELTYPE'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1::]

#Splitting Training and Test Set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=10,max_depth =10)

#Fitting model with trainig data
regressor.fit(x_train, y_train)
y_predict = regressor.predict(x_test)
# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[4,2.4,11.2,7.7,9.6,29]]))
