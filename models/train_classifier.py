import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download(['punkt','stopwords','wordnet'])
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM DisasterResponce", engine)
    X = df['message']
    Y= df.drop(['id', 'message','genre'], axis=1).astype('int')
    return  X, Y


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    pipeline = Pipeline(steps=[
     ('vect', TfidfVectorizer(tokenizer=tokenize)),
     ('multi_KN', MultiOutputClassifier(KNeighborsClassifier(), n_jobs=-1))])
    parameters = {
    'vect__norm': ['l1','l2'],
    'multi_KN__estimator__leaf_size':[10,20]
    }
    grid_obj = GridSearchCV(
    estimator=pipeline,
    param_grid=parameters,
    n_jobs=-1,
    cv=3
    )
    return  grid_obj
   


def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    i=0
    for col in list(Y_test.columns):
        print(col, classification_report(Y_test[col],y_pred[:, i]))
        i+=1


def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y= load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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