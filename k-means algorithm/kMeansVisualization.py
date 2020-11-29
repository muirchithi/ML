import numpy as np
from matplotlib import pyplot as plt
#data from test runs extracted from attached excel file
#data entry amount, time needed in seconds
data = np.array([[100,0.66],[1000,2.2],[10000,38.7],[100000,181.2],[20000,24],[30000,115],[40000,49],[50000,121],[60000,186],[70000,85],[80000,96],[90000,116]])

x,y = data.T
plt.scatter(x,y)
plt.show()
