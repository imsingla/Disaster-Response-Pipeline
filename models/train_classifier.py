import sys, pickle, re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
#nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    '''
    Fucntion to load the database from the given filepath and process them as X, y and category_names
    Input: Databased filepath
    Output: Returns the Features X & target y along with target columns names catgeory_names
    '''
    table_name = 'messages_disaster'
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table(table_name,engine)
    X = df["messages_disaster"]
    y = df.drop(["message","id","genre","original"], axis=1)
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    '''
    Function to tokenize the text messages
    Input: text
    output: cleaned tokenized text as a list object
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens



def build_model():
    '''
    Function to build a model, create pipeline, hypertuning as well as gridsearchcv
    Input: N/A
    Output: Returns the model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
        #'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100]
        #'clf__estimator__min_samples_split': [2, 4]
        } 
    cv = GridSearchCV(pipeline, param_grid=parameters)    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate a model and return the classificatio and accurancy score.
    Inputs: Model, X_test, y_test, Catgegory_names
    Outputs: Prints the Classification report & Accuracy Score
    '''
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test.values, target_names=category_names))
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))
    


def save_model(model, model_filepath):
    '''
    Function to save the model
    Input: model and the file path to save the model
    Output: save the model as pickle file in the give filepath 
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
