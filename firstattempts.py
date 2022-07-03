#This is where I will try to start coding my own basic applications of the
# machine learning content I learnt from w3schools' course

import numpy
from scipy import stats
import matplotlib.pyplot as plt

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
slope, intercept, r, p, std_err = stats.linregress(x, y)
def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)

mymodel = numpy.poly1d(numpy.polyfit(x, y, 5))
myline = numpy.linspace(min(x), max(x), 1000)
plt.plot(myline, mymodel(myline))

plt.show()
