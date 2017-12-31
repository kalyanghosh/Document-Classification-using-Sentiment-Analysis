# Document-Classification-using-Sentiment-Analysis
This repository contains code base and data files for the project titled "Document Classification using Sentiment Analysis"  

Objective: 

The main objective  of the project was to build a model which can classify documents based on the emotional and sentimental content as expressed in the document.

The steps followed to develop the model are as follows:

1. Getting the Dataset:

The dataset used for this project is the IMDb movie review dataset available at http://ai.stanford.edu/~amaas/data/sentiment/
This dataset contains a total of 50K reviews.The Training and Test set dividon can be accoring to the user's own judgement.

2. def preprocess_Data(path):

This method takes in the path where the unzipped datafile is kept in the local computer.This method randomizes the data after reading it into a Pandas dataframe so that the randomness is maintained in the data after Train/Test split of data

3. def preprocessor(text):

This method removes unwanted characters for the 'reviews' column of the dataset.Unwanted characters include HTML tags and special characters, but exclamations and emoticons are preserved because they are a potent indicator to the sentiment expressed in the review.

4. def tokenizer(text):
    
This method splits the review string into word tokens.

5. def tokenizer_porter(text):

From the Natural Language Toolkit (NLTK), we import the PorterStemmer class, that stems the words to its root word.
This method implements this functionality.

We also import the stopwords from the NLTK package to create a bag of words to include only those words which are significant in capturing the sentiment of the review.

6. Train/Test data split: 

The entire data of 50k is Test and Training data in the rati of 50:50 each(25k each)

7. Cross Validation:

A 5 fold cross validation is performed on the hyperparameter space to find the optimum set of hyperparameters

8. Model Performance Testing: 

The learned model is fitted to the Test Data to obtain the predictions.

Note: The model obtains a CV accuracy of 89% and it achieves a model performance of 90% on the Testing set.



