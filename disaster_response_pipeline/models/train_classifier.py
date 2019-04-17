import sys

import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

#import joblib
import pickle

def load_data(database_filepath):
    """
    Load labelled messages from specified database. 
   
    Args: 
        database_filepath (string): file path of database.
   
    Returns:
        X (pandas series): messages.
        y (pandas dataframe): category of labelled message.
        category_names (list): names each message can be categorised as.
    """

    engine = create_engine('sqlite:///' + database_filepath)
    
    df = pd.read_sql_table('LabelledMessages', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns.tolist()
    
    return(X, y, category_names)

def tokenize(text):
    """
    Tokenises and lemmatises text.
    
    Args: 
        text (string): a message that needs to be tokenised.
    
    Returns:
        clean_tokens (list): returns a list of clean tokens.
    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return(clean_tokens)

def build_model():
    """
    Construct a multi-output random forest classifier pipeline that performs cross validation gird search.
   
    Returns:
        cv (object): scikit learn GridSearchCV object.
    """
    
    rf_clf = RandomForestClassifier()

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(rf_clf))
        ])
    
    parameters = {
             #'clf__estimator__bootstrap': [True, False],
             #'clf__estimator__max_depth': [10, 100, None],
             #'clf__estimator__max_features': ['auto', 'sqrt'],
             'clf__estimator__n_estimators': [50, 100]
                }
             
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=3, verbose=2)

    return(cv)

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Prints the precision, recall and f1-score of the multi-ouput model.
    
    Args: 
        model (): 
        X_test (pandas dataframe): messages of the test set.
        Y_test (pandas dataframe): category of test set.
        category_names (list): names each of the messages can be categorised as. 
    """
    Y_pred = model.predict(X_test)    

    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i])) 


def save_model(model, model_filepath):
    """
    Saves model as a pickled file.
    
    Args: 
        model (object): trained model.
        model_filepath (string): where the pickled file is to be saved to.
    """
    
    #joblib.dump(model, model_filepath)
    pickle.dump(model, open(model_filepath, "wb"))


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
