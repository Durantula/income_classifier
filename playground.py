import numpy as np
import pandas
from sklearn import svm
# from sklearn.linear_model import SGDClassifier
from gensim.models import Word2Vec


# import classified data
data = pandas.read_csv("./data/cleanedClassifiedData.csv", sep=",")
classifiedCategory = pandas.read_csv("./data/classifiedCategory.csv", sep=",")
unclassifiedData = pandas.read_csv("./data/cleanedUnclassifiedData.csv", sep=",")
# data = pandas.read_csv("classifiedData.csv", header = 0)


# train model on classified data

# only get the numerical data from what was read
data = data._get_numeric_data()

X = data.values
y = classifiedCategory.values.flatten()


# clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
clf = svm.SVC(gamma='scale')
clf.fit(X,y)

# output = clf.predict(unclassifiedData.values)
output = clf.predict(unclassifiedData)

a = np.asarray(output)
np.savetxt("output.csv",a,delimiter=",")

# dataset = np.loadtxt(data, delimiter=",")
