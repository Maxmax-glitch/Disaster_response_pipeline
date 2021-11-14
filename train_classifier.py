# import libraries
import sys
import re
import pickle
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet','stopwords'])

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine

def load_data(database_filepath):
    ''' load data from database & split into two dataframes X Y '''
    
    #Load data from SQL database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse.db', engine)
    
    #define category_names
    category_names = ['request', 'offer', 'aid_related', 'medical_help', 'medical_products','search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people','refugees', 'death', 'other_aid', 'infrastructure_related',       'transport', 'buildings', 'electricity', 'tools', 'hospitals','shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather','direct_report']
    
    #Load data into two dataframes   
    X = df['message']
    Y = df[category_names]

#Note: Some messages are included duplicated in the dataframe, but with different categories. To be clarified, if okay or should be removed

    return X,Y, category_names


def tokenize(text):
    '''tokenize function cleans the input by replacing URLs with a placeholder, tokenizes and lemmantizes the sentences, removes punctuation, removes stopwords'''
    
    #define regex for detection URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
       
    #Change URLs to 'urlplaceholder'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    #remove punctuation
    tokens = nltk.word_tokenize(text)
    new_tokens= [token for token in tokens if token.isalnum()]
    
    lemmatizer = nltk.WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    #lemmantize (Converting words into their dictionary forms)
    clean_tokens = []
    for tok in new_tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    clean_sentence = []

    #Remove stop words
    for c in clean_tokens:
        if c not in stop_words:
            clean_sentence.append(c)
    
    return clean_sentence

def build_model():
    '''Use Grid Search to find optimal parameters for model & pipeline'''  
        
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
        ])
        
    parameters = {
        #TEST CASE: only 'clf__min_samples_split' parameter is used in three settings.
        #Additional parameters are available and can be used in grid search, but increase training time significantly.
        'clf__min_samples_split': [2, 3, 4]
        
        #Commented out parameters Examples:
        #'clf__n_estimators': [50, 100, 200],
        #'clf__min_samples_split': [2, 3, 4]
        }
    
    model = GridSearchCV(pipeline, param_grid=parameters)
       
    return model
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''evaluate_model function evaluates model based on common evaluation scores'''
          
    #Use trained model to predict on test set
    Y_pred = model.predict(X_test)
    #Evaluate model
    evaluation = classification_report(Y_test, Y_pred, target_names = category_names)
    
    print(evaluation)
        
    return evaluation


def save_model(model, model_filepath):
    ''' save the model to given directory'''
    #filename = 'optimized_RandomForestClassifier.sav'
    #pickle.dump(model, model_filepath)
    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
        
    return


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