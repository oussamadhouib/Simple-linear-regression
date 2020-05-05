# Simple linear regression

# Importing the libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset

dataset = pd.read_csv('Salary_Data.csv')

# dimensions

dataset.shape

# les variables X et y

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# Description

dataset.describe()

# scatterplot : 'YearsExperience' vs 'Salary'

dataset.plot.scatter(x='YearsExperience',y='Salary',color='red')

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,\
                                                  test_size=1/3,\
                                                  random_state=0)

# Fitting simple linear regression to the training set

# importer la classe LinearRegression du module linear_model de la librairie Scikit-learn

from sklearn.linear_model import LinearRegression

# Création de l'objet

regressor = LinearRegression()

# Utiliser la méthode fit de la classe LinearRegerssion

regressor.fit(X_train,y_train)

# le modèle est créer sur l'ensemble d'entraînement 
# (corrélation entre les variables YearsExperience et Salary) 

# predict the test set results

# création d'un vecteur de prédiction
# (utiliser la méthode predict de la classe LinearRegression)

y_pred = regressor.predict(X_test)

# Visualising the Training set results

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs YearsExperience')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results

plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,y_pred,color='blue')
plt.title('Salary vs YearsExperience')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
