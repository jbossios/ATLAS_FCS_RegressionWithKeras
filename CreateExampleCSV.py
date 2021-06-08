####################################################
# Create example CSV with x, x+2, y=x*x
# The plan would be to setup a NN to predict y=x*x
####################################################

import os,sys,csv,ROOT
import pandas as pd
from ROOT import *

# Create DataFrame input with x,x+2,y=x^2
print('INFO: Create data')
data = []
func = ROOT.TF1("myFunc","x*x",0,1)
for x in range(0,10000):
  x = func.GetRandom()
  data.append([x,x+2,x*x])
dataset = pd.DataFrame(data, columns=['x','x+2','y'])

# Create CSV
print('INFO: Create CSV file')
dataset.to_csv("TestData.csv",index=False)

print('>>> ALL DONE <<<')
