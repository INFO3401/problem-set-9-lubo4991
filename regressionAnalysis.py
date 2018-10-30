#Lucas Bouchard



import pandas as pd
import csv
import parsers



#Problem 1
def _init_(self, type):
    self.type=type
    self.data=[]
    
    file = open(in_file, encoding = 'utf-8')
    
def parseFile(self, filename):
    if (self.type =="csv"):
        reader=csv.reader(open('candy-data.csv'))
        for row in reader:
            self.data.append(row)
        else:
            self.data=open('candy-data.csv').read()



            

    

