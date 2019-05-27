import numpy as np
import pandas
from sklearn import svm
# from sklearn.linear_model import SGDClassifier
from gensim.models import Word2Vec
from sklearn.neural_network import MLPClassifier

# TODO LIST
# Add neural net model - DONE
# add date -- didn't improve 
# add embedded word vectors 
# add synthetic data 


# import classified data
data = pandas.read_csv("./data/cleanedClassifiedData.csv", sep=",")
classifiedCategory = pandas.read_csv("./data/classifiedCategory.csv", sep=",")
unclassifiedData = pandas.read_csv("./data/unclassifiedCleaned.csv", sep=",")
# data = pandas.read_csv("classifiedData.csv", header = 0)


# train model on classified data

# only get the numerical data from what was read
data = data._get_numeric_data()
print data


y = classifiedCategory.values.flatten()


# clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
clf = svm.SVC(gamma='scale')
# clf = MLPClassifier()
clf.fit(X,y)

# output = clf.predict(unclassifiedData.values)
output = clf.predict(unclassifiedData)

a = np.asarray(output)
np.savetxt("output.csv",a,delimiter=",")

# dataset = np.loadtxt(data, delimiter=",")
