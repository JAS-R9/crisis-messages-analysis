''' This program will use a set of existing messages stored in a database to develop a ML model
    based on how the messages are classified. Then the  model will be stored in a pickle file
    that should be specified in the arguments. 
    
    Parameters:
    argument 1: Database file name
    arugment 2: Pickle file name
    
    Returns:
    GridSearchCV: trained model 

'''


import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(database_filepath):
    ''' Loads data from the database and split messages and categoreis
    
    Parameters:
        database_filepath: The file path of the database to load dataframe
    
    Returns:
        X: dataframe of the messages
        Y: dataframe of the categories
        category_names: the names of the categories columns
    '''
    
    #Create an engine and load database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("MessagesTable2",engine)
    
    #X is the messages, Y is the categories
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = Y.columns
    
    return X,Y,category_names
    
def tokenize(text):
    ''' takes a text  and normalize,lemmatize, and tokenize it.
    
    Parameters:
        text: the text string
    
    Returns:
        tokens: List of the tokenized lemmatize and normalized
    '''
    
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())
    stop_words = stopwords.words("english")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]
    return tokens


def build_model():
    ''' this function builds the pipeline and gridsearchcv for the model
    
    Returns:
        cv: GridSearchCV using pipeline and parameters
    '''
    #build the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # I tried multiple parameters for the GridSearchCV 
    # to shorten the time some of those are commented out since GridSearch took forever to finish
    parameters = {
        'vect__max_df':(0.75,1),
#        'vect__max_features':(None,2000),
#         'vect__ngram_range': ((1, 1), (1, 2)),
#         'tfidf__use_idf': (True),
#         'tfidf__norm': ('l1', 'l2'),
#         'clf__n_jobs':(1,2),
#         'clf__estimator__n_estimators': [50, 100],
#         'clf__estimator__n_estimators': [75, 100],
#         'clf__estimator__min_samples_split': [2, 4],
#         'clf__estimator__criterion':('gini', 'entropy'),    
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


# method to convert the predected array to dataframe    
def convert_to_df(Y_pred,columns):
    ''' convert an array to dataframe 
    
    Parameters:
        Y_pred: predicted values in array format
        columns: names of the columns
    
    Returns:
        df: reutns a dataframe constructed from the predicted array
    '''
    df = pd.DataFrame(Y_pred)
    df.columns = columns
    return df


def evaluate_model(model, X_test, Y_test, category_names):
    '''evaluate the model and pring out the resutls of the classificaiton
    
    Parameters:
        model: the trained model
        X_test: the messages to classify
        Y_test: the actual results of the classificaiton 
        category_names: the names of the categories
    '''
    Y_pred = model.predict(X_test)
    Y_pred_df = convert_to_df(Y_pred,category_names)
    for column in Y_test.columns:
        names= [column+"-0",column+"-1",column+"-2"]
        print(classification_report(np.array(Y_test[column]), np.array(Y_pred_df[column]),target_names=names))


def save_model(model, model_filepath):
    '''Save the model inot a pickle file
    Parametesr:
        model: The model to save
        model_filepath: the path to save the pickle file in and the name of the file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))




def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()