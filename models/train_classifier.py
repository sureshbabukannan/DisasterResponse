# Imports
import sys
import platform

import sqlalchemy as sql
import numpy as np
import pandas as pd
from pprint import pprint
from time import time
import logging

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pickle 


# Load data from SQL to dataframe

def load_data(database_filepath):
    '''
        INPUT      : Takes a datafile name and relative path as parameter
        OUTPUT     : Pandas dataframe
       
        PROCESSING : 1. create a Database connection object    
                     2. Read SQL table into a pandas dataframe
                     3. retrun the dataframe
    '''
    dbpath = 'sqlite:///' + database_filepath
    print("Database path is ", dbpath)
    engine = sql.create_engine(dbpath)
    conn   = engine.connect()
    
    table_name = get_table_name(database_filepath)
    print("Table name is ", table_name)
    df = pd.read_sql_table(table_name, conn)
    
    # Clean dataframe remove na values
    df.dropna(axis='index',inplace=True)
    
    #split test and train data
    X, Y = split_features(df)
    
    return X, Y, Y.columns
    

def split_features(df):
    '''
        The data is split into test and train data
    '''
    
    df.message = df.message.apply(CovertBtoUtf8)
    X = pd.DataFrame(df.message)
    Y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 
       'search_and_rescue', 'security', 'military', 'child_alone', 
       'water', 'food', 'shelter', 'clothing', 'money', 
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 
       'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 
       'other_infrastructure', 'weather_related', 'floods', 'storm', 
       'fire', 'earthquake', 'cold', 'other_weather', 'direct_report'
    ]]
    
    for y_col in Y.columns:
        Y[y_col] = Y[y_col].astype(int)
    return X, Y
    

def CovertBtoUtf8(x):
    '''
        Convert bytes to UTF8 charset. This will enable the data in 
        message usable as UTF8 charset
    '''
    if isinstance(x, bytes):
        return x.decode('utf-8')
    else:
        return x     

        
def get_table_name(dbpath):
    ''' 
        Parse as windows path or Linux path
         
        INPUT      : Takes a datafile name and relative path as parameter
        OUTPUT     : lastnode of the path, which is the tablename
       
        PROCESSING : Return the table name parsed from database file path
                     can handle windows and linux style pathnames
    '''    
    
    if "\\" in dbpath :
    #if dbpath.find('\') > -1:
        return dbpath.split('\\')[-1].split(".")[0] 
         
    elif "/" in dbpath : 
        return dbpath.split('/')[-1].split(".")[0]
        
    else :
        return dbpath.split(".")[0]
    
        
def tokenize(text):     
    '''
        INPUT      : Takes a sentences from the corpus 
        OUTPUT     : list of tokens from the sentence 
       
        PROCESSING : parse the sentence with nltk word tokeniser
                     parse again the wordnet lemmatize
                     remove stop words
                     make the tokens into lower case and remove spaces
                     return the cleaned tokes as a list
    '''  

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    stop_words = stopwords.words("english")
    tokens = [t for t in tokens if t not in stop_words]
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Buils model using a random forest classifier passed to a 
# Multi Output classifier
def build_model():
    '''
        Define the model using pipeline
       
        step 1 - CountVectorizer using tokenize function
        step 2 - TfidfTransformer
        step 3 - MultiOutputClassifier with RandomForestClassifier as
                 estimator
                 
        step 4 - define the hyper parameters for the pipeline steps
        step 5 - define GridsearchCV with the parameters 
    '''
    rfc = RandomForestClassifier(n_jobs=20,
                                 n_estimators=10,
    )

    pipeline = Pipeline([
                ('vect',CountVectorizer(tokenizer=tokenize)),
                ('tfidf',TfidfTransformer()),  
                ('clf', MultiOutputClassifier(estimator=rfc))   
            ])
            
    # parameter used based on best parameter returned for 
    # RandomizedSearchCV. These were the original set of parameters
    # used for RandomizedSearchCV
    #parameters = {
    #                'vect__max_df'     : (0.5, 0.75, 1.0),
    #                'vect__ngram_range': ((1, 1),(1, 2)),  # unigrams or bigrams
    #                'tfidf__use_idf': (True, False),
    #                'tfidf__norm': ('l1', 'l2'),
    #                'clf__estimator__bootstrap': [True],
    #                'clf__estimator__max_depth': [80, 90, 100, 110],
    #                'clf__estimator__max_features': [2, 3],
    #                'clf__estimator__min_samples_leaf': [3, 4, 5],
    #                'clf__estimator__min_samples_split': [8, 10, 12],
    #                'clf__estimator__n_estimators': [100, 200, 300, 1000]
    #            }  
    
    parameters = {
                    #'vect__max_df'     : (0.75, 1.0),
                    #'vect__min_df'     : (0.001, 0.002),
                    #'vect__ngram_range': ((1, 1),(1, 2)),  # unigrams or bigrams
                    #'tfidf__use_idf': (True, False),
                    #'tfidf__norm': ('l2'),
                    'clf__estimator__bootstrap': [True],
                    'clf__estimator__max_depth': [100],
                    'clf__estimator__max_features': [2],
                    'clf__estimator__min_samples_leaf': [4],
                    'clf__estimator__min_samples_split': [10],
                    'clf__estimator__n_estimators': [1000]
    }
   
    
    model = GridSearchCV(pipeline, parameters, verbose=1)
    #model = RandomizedSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, random_state=111, n_iter=20)
        
    return model, pipeline, parameters
 

def evaluate_model(model, X_test, Y_test, category_names):
    '''
        INPUT : model object, X and Y test data, category names
        Test model using test data
       
        step 1 - predict using model for test data 
        step 2 - Display testing results
    '''
    Y_pred = model.predict(X_test.values.ravel())
          
    display_results(Y_test, Y_pred, model)
    
def display_results(Y_test, Y_pred, model):
    '''
        INPUT : Y predicted array, Y test value array, model object
        OUTPUT: Display of results using the model
    '''    
    print("\nModel):", model)
    for i,y in enumerate(Y_test.columns):
        truth = Y_test[y].astype(np.float32)
        pred = Y_pred[:,i].astype(np.float32)
        print('\n Classification report of label# {} - "{}"'.format(i+1,y))
        title_text = "Target ~ '" + y + "'"
        #print(classification_report(truth, pred, target_names=[title_text]), labels=[y]);
        print(classification_report(truth, pred));     
        print('\n Accuracy Score for target ~ "{}" : {}'.format(y, accuracy_score(truth, pred)))

def save_model(model, model_filepath):
    ''' 
        INPUT : Model object, model file name and file path 
        OUTPUT: Pickle file of the model
    '''   
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 )
        
        print('Building model...')
        model, pipeline, parameters = build_model()
        
        print('Training model...')
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        pprint(parameters)
        t0 = time()
        print("Start time : %0.3fs" % (t0))
        
        model.fit(X_train.values.ravel(), Y_train)
        #model.fit(X_train, Y_train)
        
        print("done in %0.3fs" % (time() - t0))
        print()
        
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