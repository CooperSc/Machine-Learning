import sys
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
#nltk.download('stopwords')
#nltk.download('punkt')
import heapq
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()





def loadData(path):

  '''
  reads data from the folders
  x_train : [review1, review2, ....., review_n], where each review1 is a list of tokens
  vocabulary is a dictionary: (key: word, value: count)
  '''
  # --------your code here-------------
  x_train = [[]];
  y_train = [];
  x_test = [[]];
  y_test = [];
  vocabulary = {};
  os.chdir(path)
  for sets in os.listdir():
      if not sets.startswith("."):
          os.chdir(path + "/" + sets)
          train = False
          if sets == "training_set":
              train = True
          for folder in os.listdir():
              if not folder.startswith("."):
                  os.chdir(path + "/" + sets + "/" + folder)
                  label = 0
                  if folder == "pos":
                      label = 1
                  for file in os.listdir():
                      if file.endswith('.txt'):
                          word_tokens = tokenizer.tokenize(open(file).read())
                          words = []
                          for w in word_tokens:
                            if w not in stop_words:
                                stemmedWord = ps.stem(w)
                                words.append(stemmedWord)
                                if stemmedWord in vocabulary:
                                    vocabulary[stemmedWord] = vocabulary[stemmedWord] + 1
                                else:
                                    vocabulary[stemmedWord] = 1
                          if (train):
                              y_train.append(label)
                              x_train.append(words)
                          else:
                              y_test.append(label)
                              x_test.append(words)
  return x_train, x_test, y_train, y_test, vocabulary
def getBOWRepresentation(x_train, x_test, vocabulary):
    '''
    converts into Bag of Words representation
    each column is a feature(unique word) from the vocabulary
    x_train_bow : a numpy array with bag of words representation
    '''
    # --------your code here-------------
    x_train_bow = np.zeros((len(x_train),len(vocabulary) + 1))
    x_test_bow = np.zeros((len(x_test), len(vocabulary) + 1))
    indexDict = {}
    count = 0
    for word in vocabulary:
        indexDict[word] = count
        count = count + 1
    count = 0
    for list in x_train:
        for word in list:
            if word in vocabulary:
                x_train_bow[count,indexDict[word]] = x_train_bow[count,indexDict[word]] + 1
            else:
                x_train_bow[count, 100] = x_train_bow[count, 100] + 1
        count = count + 1
    count = 0
    for list in x_test:
        for word in list:
            if word in vocabulary:
                x_test_bow[count,indexDict[word]] = x_test_bow[count,indexDict[word]] + 1
            else:
                x_test_bow[count, 100] = x_test_bow[count, 100] + 1
        count = count + 1
    return x_train_bow[:,:], x_test_bow[:,:]

def naiveBayesMulFeature_train(Xtrain, ytrain):
  # --------your code here-------------
  #Xtrain = np.where(Xtrain > 0,1,0)
  sumsPos = np.sum(Xtrain[0:700],axis=0)
  sumsNeg = np.sum(Xtrain[700:1400], axis=0)
  thetaPos = sumsPos
  thetaPos = (1 + thetaPos) / (np.sum(thetaPos))
  thetaNeg = sumsNeg
  thetaNeg = (1 + thetaNeg) / (np.sum(thetaNeg))
  return thetaPos, thetaNeg

def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg):
 # --------your code here-------------
  Xtest = Xtest[0:-1]
  yPredict = np.where(np.dot(Xtest,np.log(thetaPos)) > np.dot(Xtest,np.log(thetaNeg)),0,1)
  ytest = np.asarray(ytest)
  Accuracy = np.sum(np.where(ytest == yPredict,1,0)) / 600
  return yPredict, Accuracy

def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
  # --------your code here-------------
  model = MultinomialNB().fit(Xtrain[0:-1],ytrain)
  Accuracy = model.score(Xtest[0:-1],ytest)
  return Accuracy

def naiveBayesBernFeature_train(Xtrain, ytrain):
    # --------your code here-------------
    Xtrain = np.where(Xtrain > 0,1,0)
    sumsPos = np.sum(Xtrain[0:700], axis=0)
    sumsNeg = np.sum(Xtrain[700:1400], axis=0)
    thetaPosTrue = sumsPos
    thetaPosTrue = (1 + thetaPosTrue) / 702
    thetaNegTrue = sumsNeg
    thetaNegTrue = (1 + thetaNegTrue) / 702
    return thetaPosTrue, thetaNegTrue

def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
  # --------your code here-------------
  XtestTrue = np.where(Xtest[0:-1] > 0,1,0)
  XtestFalse = np.where(Xtest[0:-1] == 0,1,0)
  yPredict = np.where((np.dot(XtestTrue, np.log(thetaPosTrue)) + (np.dot(XtestFalse, np.log(np.ones(101) - thetaPosTrue)))) > (np.dot(XtestTrue, np.log(thetaNegTrue)) + (np.dot(XtestFalse, np.log(np.ones(101) - thetaNegTrue)))), 0, 1)
  ytest = np.asarray(ytest)
  Accuracy = np.sum(np.where(ytest == yPredict, 1, 0)) / 600
  return yPredict, Accuracy


if __name__=="__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python naiveBayes.py dataSetPath")
        sys.exit()

    print("--------------------")
    textDataSetsDirectoryFullPath = sys.argv[1]
    # os.chdir(os.getcwd() + "/data_sets")
    # textDataSetsDirectoryFullPath = os.getcwd()


    # read the data and build vocabulary from training data
    # XtrainText is a list of lists: each document is represented by a list of tokens, this
    # function should include the stemming, preprocessing etc.
    # remember to add a UNK to represent out of vocabulary terms
    XtrainText, XtestText, ytrain, ytest, vocabulary = loadData(textDataSetsDirectoryFullPath)


    # let's look at the vocab 

    print("number of unique words: ", len(vocabulary))
    print("the most common 10 words were:", heapq.nlargest(10, vocabulary, key=vocabulary.get))
    print("the least common 10 words were:", heapq.nsmallest(10, vocabulary, key=vocabulary.get))
    vocabulary = dict((word, index) for word, index in vocabulary.items() if \
                    vocabulary[word]>=3 and word in heapq.nlargest(100, vocabulary, key=vocabulary.get))
    
    print("number of unique words in vocabulary: ", len(vocabulary))




    # get BOW representation in the form of numpy arrays
    
    Xtrain, Xtest = getBOWRepresentation(XtrainText, XtestText, vocabulary=vocabulary.keys()) 
        
    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    
    print("--------------------")
    print("thetaPos =", thetaPos)
    print("thetaNeg =", thetaNeg)
    print("--------------------")

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print("MNBC classification accuracy =", Accuracy)

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print("Sklearn MultinomialNB accuracy =", Accuracy_sk)


    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print("thetaPosTrue =", thetaPosTrue)
    print("thetaNegTrue =", thetaNegTrue)
    
    print("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print("BNBC classification accuracy =", Accuracy)
    print("--------------------")
    print("--------------------")
