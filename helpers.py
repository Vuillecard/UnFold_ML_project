import pandas as pd
import numpy as np
from pandas.compat import StringIO

"""
#######################################################function description##############################################################
load_data: load the data from a dat file, as a list.
######################################################################################################################################### """

def load_data(path,n,cl=True):
    data=[]
    for i in range (n):   
        if cl:
            f= np.genfromtxt(path +"%04d" % (i+1)+'.dat',delimiter=" ",skip_header=1, usecols=[1,2])
            data.append(f)
        else:
            f= np.genfromtxt(path +"%04d" % (i+1)+'.dat',delimiter=" ",skip_header=1, usecols=[1,3])
            data.append(f)
        
    data=np.swapaxes(data,1,2)  
    return data
