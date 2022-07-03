import pandas as pd
import re
import joblib
import time
import sklearn_crfsuite

CRF_MODEL = 'crf_ner_model.pkl'
#features for all words
def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),## is_first_capital
        'word.istitle()': word.istitle(),## Check if each word start with an upper case letter
        'word.isdigit()': word.isdigit(),## is_numeric
        'word.position()': str(i),
    }
    # Features for words that are not at the beginning of a document
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.position()': str(i),
        })
    else:
        features['BOS'] = True

    # Features for words that are not at the end of a document
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.position()': str(i),
        })
    else:
        # Indicate that it is the 'end of a document'
        features['EOS'] = True

    return features

# functions for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, postag, label) in doc]

def transform_data(trf):
    ls = []
    arr = []
    for i in range(len(trf)):
        if trf[i] != "":
            a = trf[i].split(' ')
            if len(a) != 3:
                print(a[0] + " " + a[1])
            ls.append(tuple(a))
    arr.append(ls)
    return arr

#training the model
def train_ner(X_train, y_train):
    start_train = time.time()
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',## sử dụng thuật toán LBGS -L1,L2
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    joblib.dump(crf, CRF_MODEL)
    print("Training time:", time.time() - start_train)
    
#build features and create train and test 
with open(r"/data/train_v2.txt", 'r',encoding="utf8") as f:
    trf = f.read().splitlines()
    
if __name__ == "__main__":
  train_sents = transform_data(trf)
  X_train = [extract_features(doc) for doc in train_sents]
  y_train = [get_labels(doc) for doc in train_sents]
  train_ner(X_train, y_train)
