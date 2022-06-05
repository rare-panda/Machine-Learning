import os
import re
import sys
from math import log


def read_txt_file(file_path):
    with open(file_path,'r', encoding = "ISO-8859-1") as f:
        return(f.read())

def get_words(file_path):
    txt_file_data = read_txt_file(file_path)
    words = re.sub('[^a-zA-Z]',' ', txt_file_data)
    words = words.strip().split()  
    return words

def update_vocabulary(vocabulary, words,folder_name):
    if folder_name == 'Spam': 
        folder_no = 0
    else:
        folder_no = 1
    
    for word in words:
        if not word in vocabulary:
            vocabulary[word] = [0,0]
        vocabulary[word][folder_no] += 1
    return vocabulary

def get_counts_vector(stop_words_bool):
    file_count = [0,0]
    #open stopwords file and create a list out of it:
    stop_words = open("stopWords.txt").readlines()
    
    #open spam and ham files:
    folders = ['Spam', 'Ham']
    vocabulary = {}
    for name in folders:
        folderpath = input('Enter ' + name + ' training folder path:')# folder path
        os.chdir(folderpath)    # change directoty
        if name == "Spam":  #No of files in spam and ham
            file_count[0] = len(os.listdir(folderpath))
        else:            
            file_count[1] = len(os.listdir(folderpath))
        #iterate through each file
        for file in os.listdir():
            words = []
            file_path = f"{folderpath}/{file}"
            words = get_words(file_path) #Get words in file
            vocabulary = update_vocabulary(vocabulary, words,name)#Update vocabulary
    if stop_words_bool == False: #if stop words are to be removed, remove them
        for word in stop_words:
            if word.strip() in vocabulary:
                vocabulary.pop(word.strip())
    return vocabulary,file_count

def find_accuracy(vocabulary,file_count):
    no_words = [0,0] # no of words in vocabulary in each spam and ham folders
    accuracy_count = 0
    # find total words in spam and ham
    for key in vocabulary:
        no_words[0] += vocabulary[key][0]
        no_words[1] += vocabulary[key][1]

    #Test data folder
    folders = ['Spam', 'Ham']
    Test_file_count = 0
    for name in folders:
        folderpath = input('Enter ' + name + ' testing folder path:')# folder path
        os.chdir(folderpath)    # change directoty
        total_Folders = 0
        Test_file_count += len(os.listdir(folderpath))
        #iterate through each file
        for file in os.listdir():
            total_Folders += 1
            probs_class = []
            words = []
            file_path = f"{folderpath}/{file}"
            words = get_words(file_path)    #words in file
            
            for folder in range (1,2):  # for spam , ham cal; prob of each word in file
                total_prob = 0
                for word in words:
                    if word in vocabulary:
                        prob_word = (vocabulary[word][folder-1] + 1) / (no_words[folder-1] + len(vocabulary)) #prob(word|folder)
                        total_prob += log(prob_word)
                total_prob += log(file_count[folder -1] / sum(file_count))
                probs_class.append(total_prob * (-1))
            
            max_value = max(probs_class)
            max_index = probs_class.index(max_value)
            
            if max_index == (folder-1): 
                accuracy_count += 1
    return (float)(accuracy_count / Test_file_count)*100

    
if __name__ == "__main__":
    # check if user wants to consider stop words or no
    stop_words_val = input('Do you wont to consider stop words - yes/no: ')
    if stop_words_val == 'yes':
        stop_words_bool = True
    elif stop_words_val == 'no':
        stop_words_bool = False
    else:
        print('Run the file again and enter a valid argument')
        sys.exit()

    #Get counts vector
    counts = []
    counts,file_count = get_counts_vector(stop_words_bool)
    
    #Test 
    accuracy = find_accuracy(counts,file_count)
    print('Accuracy is:' + str(accuracy))
    