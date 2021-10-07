# import libraries
import nltk
import re
import pandas as pd

import sys
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# Necessary steps for nltk:
# Download necessary NLTK data
nltk.download(['punkt', 'wordnet'])

# Define expression to detect a URL
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    '''The load_data function loads data from an inputed database filepath.
    
    It will extract the variable (X) and the dependent variables (Y) respectively.   
    
    INPUT:
    database_filename (str): The name of the database where the data is stored in SQLite
    
    OUTPUT:
    X: Pandas dataframe containing messages
    Y: Pandas dataframe containing categories of the disaster
    category_names: Names of categories (list)
    '''
    
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("Disaster_Response", engine)
    
    # Extract X and Y
    # Message will be input column
    X = df["message"]
    
    # Keeping only the output category variables, so dropping everything else
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    category_names  = Y.columns.tolist()
    return X, Y, category_names 


def tokenize(text):
    ''' The tokenize functionbreaks the texts into tokens, replaces URLs 
        as well as lemmatizes the tokens. 

    INPUT:
    text: sentences
    
    OUTPUT:
    clean_tokens: list of generated tokens
    '''
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer() 

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    ''' The build_model builds the pipeline for training the model. 
        It will use a pipeline object with transformers and estimators (CounterVectorizer,
        TfidfTransformer and MultiOutputClassifier) 
    
    This uses MultiOutputClassifier to run the RandomForestClassifier as the dependent variable (Y)
    has multiple categorical columns, so this will be a better fit
    
    To build the model, this function will also use GridSearchCV to optimize hyper parametrs of 
    the model
      
    OUTPUT:
    cv: Finished model after GridSearch
    
    '''
    
    #Define the model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    #Specify parameters for the GridSearchCV
    #have chosen less parameters than normally would for speed
    parameters = {
        'clf__estimator__n_estimators': [10, 20,30],
    }

    #Return the GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
  


def evaluate_model(model, X_test, Y_test, category_names):
    '''The evaluate_model function evaluates the model on the test data set.
    It does this by iterating through columns and calling sklearn's
    classification report on each (showing f1 score, precision and recall).
    
        
    INPUT:
    model: The predictor Machine Learning Model
    X_test: Test data set of X
    Y_test: Test data set of Y
    category_names (list): List of category names)
    
    OUTPUT:
    classification_report for each column
    '''
    # Predict test data
    Y_pred = model.predict(X_test)
    
    # Print a classification report for each column 
    for i in range(len(category_names)):
        print("Report for target column: " + category_names[i])
        print(classification_report(Y_test[category_names[i]], Y_pred[:, i]))


def save_model(model, model_filepath):
    '''The save_model function will save the model in a pickle file.
    
    INPUT: 
    model: Machine Learning model for predicting
    model_filepath(str): The path where the pickle file will save
    
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
