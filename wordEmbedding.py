import numpy as np
import pandas
# from gensim.models import Word2Vec
import gensim


# Load Google's pre-trained Word2Vec model.
# model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)  


def avg_word_vector(word):
  # word_vectors = model.word_vec(word)
  word_vectors = [10,12,134,13]
  return np.mean(word_vectors)


def average_sentence_vector(sentence):
  avg_sentence_vector = []
  for word in sentence:
    # if word in model.vocab:
    word_vector = avg_word_vector(word)
    avg_sentence_vector.append(word_vector)
  return np.mean(avg_sentence_vector)

def write_row_to_csv(row): 
  with open('unclassifiedDataW2V.csv','wb') as file:
    file.write(row)


#get data for the model 
data = pandas.read_csv("unclassifiedData.csv", usecols=['description'])

for row in data.values:
  sentence = row[0].lower().split()
  sentence_vec = average_sentence_vector(sentence)
  row = np.append(row, sentence_vec)
  write_row_to_csv(row)
  

