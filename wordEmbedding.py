import numpy as np
import pandas
# from gensim.models import Word2Vec
import gensim


# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)  


def avg_word_vector(word):
  word_vectors = model.word_vec(word)
  return np.mean(word_vectors)


def average_sentence_vector(sentence):
  avg_sentence_vector = []
  for word in sentence:
    if word in model.vocab:
      word_vector = avg_word_vector(word)
      avg_sentence_vector.append(word_vector)
  return np.mean(avg_sentence_vector)

def write_to_csv(row): 
  with open('foo.csv', 'wb') as abc:
    np.savetxt(abc, row, delimiter=",", fmt='%s')


#get data for the model 
data = pandas.read_csv("unclassifiedData.csv", usecols=['description'])
updated_data = []
for row in data.to_numpy():
  sentence = row[0].lower().replace(",",'').split()
  sentence_vec = average_sentence_vector(sentence)
  row[0] = row[0].replace(",","")
  row = np.append(row, sentence_vec)
  updated_data.append(row)

write_to_csv(updated_data)
  

