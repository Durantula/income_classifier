import numpy as np
import pandas
# from gensim.models import Word2Vec
import gensim
import csv


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

def write_to_csv(data): 
  with open('word_data.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(data)
  csvFile.close()


#get data for the model 
data = pandas.read_csv("./data/original_misc.csv", usecols=['amount','description'])
updated_data = []
for row in data.values:
  sentence = row[1].lower().replace("-", " ").split()
  sentence_vec = average_sentence_vector(sentence)
  row = np.append(row, sentence_vec)
  updated_data.append([row[0],row[1],row[2],row[3]])

write_to_csv(updated_data)
  

