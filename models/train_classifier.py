import sys
import numpy
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import TruncatedSVD
from utils import tokenize

def load_data(database_filepath):
    '''
    Load Predictor, Response, and Category Names from sqlite database
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('SELECT * FROM disaster_response', engine)

    category_names = list(df.columns[2:])
    X = df['message'].values
    Y = df.drop(['message', 'genre'], axis=1).values

    return X, Y, category_names

def build_model():
    '''
    Build Machine Learning Pipeline Model and Use Grid Search to find the optimal parameters
    '''
    pipeline = Pipeline([('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('dim_red', TruncatedSVD(350)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1,
            class_weight='balanced')))])

    parameters = {'dim_red__n_components': [150, 250, 350],
        'clf__estimator__min_samples_leaf': [75, 100, 150]}

    model = GridSearchCV(pipeline, parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model on test data
    '''
    Y_pred = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        print(category_names[i])
        print(classification_report(Y_test[:, i], Y_pred[:, i], zero_division=0))
    


def save_model(model, model_filepath):
    '''
    Save the model as pickle file
    '''
    joblib.dump(model, model_filepath, compress=3)
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()