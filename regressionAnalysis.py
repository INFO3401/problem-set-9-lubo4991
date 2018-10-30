#Lucas Bouchard



import pandas as pd
import csv
import parsers
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogistictRegression

import matplotlib.pyplot as plt  
#%matplotlib inline


# PROBLEM 1. Using the candy-data.csv file in the repo, populate an AnalysisData object that will hold the data you'll use for today's problem set. You should read in the data from the CSV, store the data in the dataset variable, and initialize the xs (column name) and targetY variables appropriately. targetY should reference the variable describing whether or not a candy is chocolate.

def _init_(self, type):
    self.type=type
    self.data=[]
    return
_init_()
    
def parseFile(self, filename):
    file = open(filename, encoding = 'utf-8')
    if (self.type =="csv"):
        reader=csv.reader(open(filename))
        for row in reader:
            self.data.append(row)
        else:
            self.data=open(filename).read()
    return
parseFile(self, 'candy-data.csv')

#PROBLEM 2. Create a function to initialize a LinearAnalysis object that takes a targetY as its input parameter. Create the same function for LogisticAnalysis. Note that you will use the LinearAnalysis object to try to predict the amount of sugar in the candy and the LogisticAnalysis object to predict whether or not the candy is chocolate.

#dataset.plot(x='competitorname', y='sugarpercent', style='o')  
#plt.title('candy name vs. sugarpercent')  
#plt.xlabel('candy name')  
#plt.ylabel('sugarpercent')  
#plt.show()  


def linearAnalysis_Object(filename, self):
     df = pd.read_csv(filename, encoding='latin1')
    #prepping the data
     X = df[[:,1:9]]
     Y = df[[:, 10]]    #is this where we would fit target Y and target X?
        
     #X_features = df.iloc[:,1:9]
     #Y_features = df.iloc[:, 10]
    #Split data into training and test sets
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
    #Training the algorithm
     regressor = LinearRegression()  
     regressor.fit(X_train, y_train) 
    #intercept
     #print(regressor.intercept_) 
    #Slope
     #print(regressor.coef_) 
    
    #Making the prediction
     y_pred = regressor.predict(X_test) 
    #Compare test vs. train
     df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
     df 
    
linearAnalysis_Object('candy-data.csv', self)

    
#PROBLEM 3. Add a function to the LinearAnalysis object called runSimpleAnalysis. This function should take in an AnalysisData object as a parameter and should use this object to compute which variable best predicts how much sugar a candy contains using a linear regression. Print the variable name and the resulting fit.

#def runSimpleAnalysis():


#PROBLEM 4.Add a function to the LogisticAnalysis object called runSimpleAnalysis. This function should take in an AnalysisData object as a parameter and should use this object to compute which variable best predicts whether or not a candy is chocolate using logistic regression. Print the variable name and the resulting fit. Do the two functions find the same optimal variable? Does one outperform the other?  

#def runSimpleAnalysis():
    

