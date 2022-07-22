"""
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
matplotlib.use('TkAgg')
#opens a window with a histogram of the random numbers the more random floats the flatter the histogram
plt.hist(randomNumbers, 100)
plt.show()

#creating an array with floats with a normal distribution; mean, deviation, no. of value
normalDistribution = numpy.random.normal(5.0, 1.0, 100000)
#uses matplotlib to plot a histogram, of the array previously defined, with 100 bras
plt.hist(normalDistribution, 100)
plt.show()

# a scatter graph needs two arrays inputted of equal length
x = numpy.random.normal(100.0, 7.0, 10000)
y = numpy.random.normal(5.0, 8.0, 10000)
plt.scatter(x, y)
plt.show()


from scipy import stats
#plot a scatter graph
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

#calculate regression line
slope, intercept, r, p, std_err = stats.linregress(x, y)
#function that map's each x coordinate to a point on the regression line
#this can be used to predict values if you have the x-coordinate (remembering regression in one way)
def myfunc(x):
  return slope * x + intercept
#runs each value through the function to create a new array
mymodel = list(map(myfunc, x))

plt.scatter(x, y)
#plot a line
plt.plot(x, mymodel)
plt.show()


#polynomial regression lines
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 22, 100)
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()
#polynomial r value (called r-squared)
from sklearn.metrics import r2_score
print(r2_score(y, mymodel(x)))


#using more than one variable in regression lines
import pandas
df = pandas.read_csv("cars.csv")
#independent variables list
X = df[['Weight', 'Volume']]
#dependent variable
y = df['CO2']
from sklearn import linear_model
regr = linear_model.LinearRegression()
# .values removes the warning, I think it removes the header or names of the data used
regr.fit(X.values, y.values)
#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])
print(predictedCO2)
print(regr.coef_)


import numpy
from sklearn.metrics import r2_score
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(train_y, mymodel(train_x))

print(r2)



import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

df = pandas.read_csv("shows.csv")
#print(df)

d = {'UK': 0, 'USA': 1, 'N': 2}
#map takes a dictionary and then replaces the values in a csv
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

#print(df)

# X is the "input" and is the data we are using to predict from
# y is the target, these are the values we are trying to predict
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]

y = df['Go']
#print(X)
#print(y)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X.values, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()
"""
"""
My understanding of the decision tree (with reference to the png):
 the top value such as nationality or age in the tree
  -this is the condition that splits the set of values apart, true goes left, false goes right
 the second value (gini)
  -this refers to the "quality" of the split, where the split is in regards to the range of values
   in this case 6.5 is in the middle of 4 and 9 (the smallest and largest values)
   so the gini is almost exactly 0.5 which means it would be a perfectly even split, not sure why it isn't
   a gini of 0.0 would mean that the range is 0
 the third value is the number of samples
  -this is the number of values in the branch at this point in the tree
 the fourth is value and I'm  not sure entirely how this works or what it means
"""
"""

#using the decision tree to predict whether you should or shouldn't go to a comedian
#[Age, Experience, Rank, Nationality] not sure why the double brackets
print( dtree.predict([[40, 10, 7, 1]]) )


"""

"""
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()

#accuracy = how often the model is correct
#  (true positive + true negative)/Total predictions
Accuracy = metrics.accuracy_score(actual, predicted)

#precision = how many of the predicted positives are actually positive
#  true positive/(true positive + false positive)
Precision = metrics.precision_score(actual, predicted)

#sensitivity = how many actual positives are predicted correctly
#  true positive/(true positive + false negative)
Sensitivity_recall = metrics.recall_score(actual, predicted)

#specificity = sensitivity for negative results; how many actual negative results are predicted negative
#  true negative/(true negative + false positive)
Specificity = metrics.recall_score(actual, predicted, pos_label=0)

#f-score = ""F-score is the "harmonic mean" of precision and sensitivity. It considers both false positive and false negative cases and is good for imbalanced datasets.""
# doesn't take into account true negatives
#  2*( (Precision * Sensitivity)/(Precision + Sensitivity) )
F1_score = metrics.f1_score(actual, predicted)


print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})
""" 

"""
#unsupervised learning
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y))

#drawing a dendrogram?
#the height on the y-axis in a dendrogram is the distance between two clusters of datapoints

#linkage_data = linkage(data, method='ward', metric='euclidean')
#dendrogram(linkage_data)


#not quite sure what the commands do specifically but it colour codes the data splitting it into two groups
#I assume from the dendrogram?
hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data)

plt.scatter(x, y, c=labels)

plt.show()
"""

"""
import numpy
from sklearn import linear_model

#Reshaped for Logistic function.
X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)

#predict if tumor is cancerous where the size is 3.46mm:
#as the array is only a single value it needs to be reshaped, not sure what that means though
predicted = logr.predict(numpy.array(3.46).reshape(-1,1))
print(predicted)

#this tells us that when the tumor increases by 1 (mm) the odds of it being cancerous increase by 4x
log_odds = logr.coef_ 
odds = numpy.exp(log_odds)
print(odds)#4.035...

def logit2prob(logr, X):
  log_odds = logr.coef_ * X + logr.intercept_
  odds = numpy.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)

print(logit2prob(logr, X))

print(logr.predict(numpy.array(5.00).reshape(-1,1)))
"""

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

X = iris['data']
y = iris['target']

logit = LogisticRegression(max_iter = 10000)

print(logit.fit(X,y))

print(logit.score(X,y))