from numpy import *
import os
import re
import sys
from collections import Counter
from math import log
from math import e

# Returning the list of stopwords in a file
def extract_stopwords(stopwords_file):
    stopWords = []
    fil = open(stopwords_file)
    stopWords = fil.read().strip().split()
    return stopWords

# Read data files without removing stopWords
def read_withStopWords(folder):
    files = os.listdir(folder)
    dictionar = {}
    vocabular = []
    for f in files:
        fil = open(folder + "/" + f,encoding = "ISO-8859-1")
        words = fil.read()                                      
        all_words = words.strip().split()                       
        dictionar[f] = all_words
        vocabular.extend(all_words)
    return vocabular, dictionar

# Read data files by removing stopWords
def read_withoutStopWords(folder, stopWordsF):
    files = os.listdir(folder)
    dictionar = {}
    vocabular = []
    stopWords = extract_stopwords(stopWordsF)
    for f in files:
        fil = open(folder + "/" + f, encoding = "ISO-8859-1")
        words = fil.read()                                     
        words = re.sub('[^a-zA-Z]',' ', words)     
        words_all = words.strip().split()                       
        reqd_words = []                                         
        for word in words_all:
            if(word not in stopWords):
                reqd_words.append(word)
        dictionar[f] = reqd_words
        vocabular.extend(reqd_words)
    return vocabular, dictionar

# To train for MCAP Logistic Regression and returning the weight
def LR_train(train_features, labelList, lam_da):
    featureMatrix = mat(train_features)                           
    p,q = shape(featureMatrix)
    labelMatrix = mat(labelList).transpose()
    eeta = 0.1
    weight = zeros((q,1))
    number_of_iterations = 1000
    for i in range(number_of_iterations):
        predictionar_condProb = 1.0/(1 + exp(-featureMatrix*weight))
        error = labelMatrix - predictionar_condProb
        weight = weight + eeta*featureMatrix.transpose()*error - eeta*lam_da*weight
    return weight


# Apply MCAP Logistic Regression on the given test sets and returns the Accuracy of LR
def LR_apply(weight, test_features, lengthTest_spamdictionary, lengthTest_hamdictionary):
    featureMatrix = mat(test_features)
    res = featureMatrix*weight
    val = 0
    length_all_dictionary = lengthTest_spamdictionary + lengthTest_hamdictionary
    for i in range(lengthTest_spamdictionary):
        if(res[i][0] < 0.0):
            val += 1
    i = 0
    for i in range(lengthTest_spamdictionary+1, length_all_dictionary):
        if(res[i][0] > 0.0):
            val += 1
    return (float)(val/length_all_dictionary)*100

def feature_vector(all_distinct_words, dictionar):
    feature = []
    for f in dictionar:
        row = [0]*(len(all_distinct_words))
        for i in all_distinct_words:
            if(i in dictionar[f]):
                row[all_distinct_words.index(i)] = 1
        row.insert(0,1)                                 
        feature.append(row)
    return feature                                      

if __name__ == "__main__":
    if(len(sys.argv) != 7):
       
        print("Wrong arguments passed. Send correct Arguments ")
        
        print("Correct format is:python NaiveBayes.py <TrainPath_spam> <TrainPath_ham> <TestPath_spam> <TestPath_ham>", 
                "<filePath_stopWords> <yes/no to add/remove stopWords>")
        sys.exit()

    spam_train = sys.argv[1]
    ham_train = sys.argv[2]
    spam_test = sys.argv[3]
    ham_test = sys.argv[4]
    stopWords = sys.argv[5]
    stopWords_throw = sys.argv[6]
    lam_da = 0.05

    #checking to add/remove stopWords
    if(stopWords_throw == "yes"):
        spamvocabular_train, spamdictionar_train = read_withoutStopWords(spam_train, stopWords)
        hamvocabular_train, hamdictionar_train = read_withoutStopWords(ham_train, stopWords)
    else:
        spamvocabular_train, spamdictionar_train = read_withStopWords(spam_train)
        hamvocabular_train, hamdictionar_train = read_withStopWords(ham_train)

    spamvocabular_test, test_spamdictionar = read_withStopWords(spam_test)
    hamvocabular_test, test_hamdictionar = read_withStopWords(ham_test)

    all_distinct_words = list(set(spamvocabular_train)|set(hamvocabular_train))
    all_traindictionar = spamdictionar_train.copy()
    all_traindictionar.update(hamdictionar_train)

    all_testdictionar = test_spamdictionar.copy()
    all_testdictionar.update(test_hamdictionar)

    labelList = []
    for i in range(len(spamdictionar_train)):
        labelList.append(0)
    i = 0
    for i in range(len(hamdictionar_train)):
        labelList.append(1)

    train_features = feature_vector(all_distinct_words, all_traindictionar)
    test_features = feature_vector(all_distinct_words, all_testdictionar)
    
    weight = LR_train(train_features, labelList, lam_da)
    accuracy = LR_apply(weight, test_features, len(test_spamdictionar), len(test_hamdictionar))
    print("The Accuracy of Logistic Regression is: ", accuracy)
