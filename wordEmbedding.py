import numpy as np
import pandas
from gensim.models import Word2Vec


sentences = []

#get data for the model 
data = pandas.read_csv("unclassifiedData.csv", usecols=['description'])

for row in data.values:
  sentence = row[0].lower().split()
  sentences.append(sentence)

# train model
model = Word2Vec(sentences, min_count=200)


#store model 
model.save('mymodel')


#apply model to the unsupervised version 
# model['beem']
