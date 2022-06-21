import numpy
from scipy import stats


numbers = [1,2,3,4,5,3]
print("mean: "+str(numpy.mean(numbers)) )
print("mode: "+str(stats.mode(numbers)) )
print("median: "+str(numpy.median(numbers)) )


speed = [86,87,88,86,87,85,86]
deviation = numpy.std(speed)
variance = numpy.var(speed)
print("Standard deviation of speeds: "+str(deviation))
print("Variance: "+str(variance) )


ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
percentile = numpy.percentile(ages, 55)
print("Percentile: "+str(percentile))


import matplotlib
import matplotlib.pyplot as plt
#creating an array with 100000 random floats between 0 and 5
randomNumbers = numpy.random.uniform(0.0, 5.0, 100000)
print(randomNumbers)
matplotlib.use('TkAgg')
#opens a window with a histogram of the random numbers
plt.hist(randomNumbers, 100)
plt.show()

#creating an array with floats with a normal distribution; mean, deviation, no. of value
normalDistribution = numpy.random.normal(5.0, 1.0, 100000)
#uses matplotlib to plot a histogram, of the array previously defined, with 100 bras
plt.hist(normalDistribution, 100)
plt.show()