import numpy as np
def Integral(n,bins):
 return sum(np.diff(bins)*n) 
