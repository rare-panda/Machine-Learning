Naive Bayes:
	- Run the program NaiveBayes.py
	- Program will ask user if stop words should be considered or no
	- Program will ask user path for spam training folder
	  example:C:\Users\kxc200005\Desktop\KC\CS 6375 - Machine Learning\Assignment\Assignment 3\train\spam
	- Program will ask user path for ham training folder
	  example: C:\Users\kxc200005\Desktop\KC\CS 6375 - Machine Learning\Assignment\Assignment 3\train\ham
	- Program will ask user path for spam testing folder
	- Program will ask user path for ham testing folder
	- Program will print the accuracy

Logistic Regression:
	- Make sure training and testing folders are in the same directory as the program
	- Run the program
	  python LR.py <Path of spam training files> <Path of ham training files> <Path of test spam files> <Path of ham test files> <stopWords file path> <yes/no to remove stop-words>
	  example : python LR.py train\spam train\ham test\spam test\ham stopWords.txt yes
	- Program will print the accuracy