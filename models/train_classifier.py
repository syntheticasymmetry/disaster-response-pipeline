import sys

import pandas as pd

from sqlalchemy import create_engine, inspect
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# download necessary nltk data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(database_filepath):
    """
    Load data from the specified SQLite database.

    Inputs: 
    database_filepath (str): Path to the SQLite database file containing the disaster messages data.

    Returns:
    X (pd.Series): Messages data (features).
    Y (pd.DataFrame): Categories for each message (target labels).
    category_names (pd.Index): List of category names for the target labels.
    """
    # load data from SQLite database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Messages', engine)

    # define feature and target variables
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'genre'])

    return X, Y, Y.columns


def tokenize(text):
    """
    Tokenization function to process text data.

    Inputs:
    text(str): The unput text to be tokenized and processed.

    Returns:
    list: A list of clean, processed tokens (words) from the input text.
    """
    # normalize text: lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # tokenize text and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    # lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return clean_tokens


def build_model():
    """
    Builds a machine learning pipeline with text preprocessing and a multi-output classifier, then uses a GridSearchCV to tune
    the hyperparameters:
    - Tokenizes and transforms the text data with CountVectorizer and TfidfTranformer.
    - Reduces dimensionality using TruncatedSVD.
    - Classifies multi-output labels using a RandomForestClassifier.
    
    Returns:
    GridSearchCV: A GridSearchCV object that will tune the hyperparameters in order to find the best model configuration.
    """
    # Define the pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1,2))),
                ('tfidf', TfidfTransformer()),
                ('svd', TruncatedSVD(n_components=100))
            ]))
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, min_samples_split=2)))
    ])
    # Define parameters for GridSearchCV
    parameters = {
        'clf__estimator__n_estimators': [50,100],
        'clf__estimator__min_samples_split': [2,4]
    }
    
    # Use GridSearchCV to tune parameters
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model's performance on the text set and prints out a classification report for each category.

    Inputs:
    model (Pipeline): Trained scikit-learn model to evaluate.
    X_text (pd.Series): Text set features.
    Y_test (pd.DataFrame): Test set labels.
    category_names (pd.Index): Names of the target cateories.

    Returns:
    None
    """
    # make predictions on the test set
    Y_pred = model.predict(X_test)

    # print a classification report for each category
    for i, col in enumerate(category_names):
        print(f'Category: {col}')
        print(classification_report(Y_test[col], Y_pred[:, i]))
    
    # calculate and print the overall accuracy
    accuracy = (Y_pred == Y_test.values).mean()
    print(f'Overall accuracy: {accuracy}')


def save_model(model, model_filepath):
    """
    Saves the trained model as a pickeld file.

    Inputs:
    model (Pipeline): Trained model to save.
    model_filepath (str): Filepath where the model should be saved as a .pkl file.

    Returns:
    None
    """
    # save the model as a pickle file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    Main function to train and save a multi-output classifier for disaster response.

    The function loads data from a database, splits it into training and test sets, builds a machine learning pipeline,
    trains the model, evaluates its performance and saves the trained model as a pickle file.

    Command Line Arguments:
    database_filepath (str): Filepath of the SQLite database containing disaster messages and categories.
    model_filepath (str): Filepath to save the trained model as a pickle file.

    Return:
    None
    """
    #check the correct number of command-line arguments
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # split data into training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
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