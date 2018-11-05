#Lucas Bouchard



import pandas as pd
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LogistictRegression
from sklearn.metrics import r2_score
#import matplotlib.pyplot as plt  
#%matplotlib inline

#https://pythonspot.com/linear-regression/

# PROBLEM 1. Using the candy-data.csv file in the repo, populate an AnalysisData object that will hold the data you'll use for today's problem set. You should read in the data from the CSV, store the data in the dataset variable, and initialize the xs (column name) and targetY variables appropriately. targetY should reference the variable describing whether or not a candy is chocolate.

#AnalysisData
#PART (a) AnalysisData, which will have, at a minimum, attributes called dataset (which holds the parsed dataset) and variables (which will hold a list containing the indexes for all of the variables in your data). 
class AnalysisData:

#Initialize attributes
    def __init__(self):
        self.dataset=[]
        self.X_variables=[]
    
        
    
#function that opens file and removes string columns
    def parserFile(self, filename):
        self.dataset=pd.read_csv(filename)

        #Exclude uncomparable variables from X_variables
        #self.X_variables=self.dataset[[:,1:12]]/1-9
        for variable in self.dataset.columns.values:
            if variable != "competitorname":
                self.X_variables.append(variable)
                
candy_data = AnalysisData()
candy_data.parserFile('candy-data.csv')
#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html          
#PART (b) LinearAnalysis
#Which will contain your functions for doing linear regression and have at a minimum attributes called bestX (which holds the best X predictor for your data), targetY (which holds the index to the target dependent variable), and fit (which will hold how well bestX predicts your target variable).
class LinearAnalysis:
    
    def __init__(self,target_Y):
        self.bestX =None
        self.targetY = target_Y
        self.fit=None
    
    def runSimpleAnalysis(self, data):
        linear_r2=-1
        best_linear_variable=None
        #establish independent variable
        for column in data.X_variables:
            if column != self.targetY:
                Y_variable= data.dataset[column].values
                Y_variable=Y_variable.reshape(len(Y_variable),1)
                #Regression 
                regression = LinearRegression()
                regression.fit(Y_variable, data.dataset[self.targetY])
                r_score = regression.predict(Y_variable)
                r_score = r2_score(data.dataset[self.targetY],Y_variable)
                if r_score > linear_r2:
                    linear_r2 = r_score
                    best_linear_variable = column
        self.bestX = best_linear_variable
        print(best_linear_variable, linear_r2)
        
        
        

#Part C
class LogisticAnalysis:
    
    def __init__(self, target_Y):
        self.bestX = None
        self.targetY = target_Y
        self.fit = None
    def runSimpleAnalysis2(self, data):
        r2=-1
        best_variable=None
        #establish independent variable
        for column in data.X_variables:
            if column != self.targetY:
                Y_variable= data.dataset[column].values
                Y_variable=Y_variable.reshape(len(Y_variable),1)
                #Regression 
                regression = LogisticRegression()
                regression.fit(Y_variable, data.dataset[self.targetY])
                r_score = regression.predict(Y_variable)
                r_score = r2_score(data.dataset[self.targetY],Y_variable)
                if r_score > r2:
                    r2 = r_score
                    best_variable = column
        self.bestX = best_variable
        print(best_variable, r2)

#Problem 1
#candy_data = AnalysisData()
#candy_data.parserFile('candy-data.csv')

#PROBLEM 2. Create a function to initialize a LinearAnalysis object that takes a targetY as its input parameter. Create the same function for LogisticAnalysis. Note that you will use the LinearAnalysis object to try to predict the amount of sugar in the candy and the LogisticAnalysis object to predict whether or not the candy is chocolate.
#ABOVE

#Problem 3
candy_data_lin_analysis = LinearAnalysis('sugarpercent')
candy_data_lin_analysis.runSimpleAnalysis(candy_data)



     
        
        
        
        
        
        

    

