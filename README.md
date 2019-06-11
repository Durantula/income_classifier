# income_classifier
Final Year Project for UniMelb Masters

Author: Matthew Duran 

INCOME CLASSIFICATION 

How to run? 
To run the main income classifier, type into your command prompt "python income_classifier.py"

The results of script will be stored in the root directory as "output.csv"

WORD EMBEDDING 

How to run? 

Note: you will need to download the word2vec model first as indicated in NOTE 2 below and store it in the folder /model

To run the word embedding script type 
`python word_embedding_calculation"

The output will be printed to word_data.csv in the root directory


1.	Both initial and final results of the algorithms can be accessed in the csv file “Income Classification Results.csv”. A income transaction is classified as 1 and other transaction types are classified as 0. 
2.	The pretrained GoogleNews-vectors-negative300.bin.gz model has been excluded from submission of documents due to it’s large size. The file is 1.5gb. To implement the word embedding script you will need to clone the following git repository that contains the file https://github.com/mmihaltz/word2vec-GoogleNews-vectors. Once accessed, extract and add the model to the /model directory. 
3.	The machine learning script that creates and evaluates the model is accessible as a python script “income_classifier.py”.  Comments indicate how to switch between MLP and SVM
4.	The word embedding script is titled ‘word_embedding_calculation.py’. It was used to transform string descriptions into single dimension word embeddings. 
5.	The data file includes the classified training data and the unclassified data for evaluation
  - ClassifiedInput.csv – classified data
  - classifiedCategory.csv – the classification category 
  - classifiedOriginal.csv – original classified data source
  - evaluation_classification_data.csv – evaluation data
  - evaluationOriginal.csv – original evaluation data
