# Disaster Response Pipeline Project

## Project Overview
In this project we build a machine learning pipeline to categorize disaster messages. These messages are classified into multiple categories to ensure that the appropriate disaster relief agencies receive them.<br>
The project includes an **ETL pipeline** to clean and store the data and an **ML pipeline** to train and evaluate the model.<br>
Finally, a **Flask web app** is built to enable users to classify new messages.

## Instructions:
1. **Set up the database and model** by running the following commands in the project's root directory.

    - **Run the ETL pipeline** to clean the data and store it in a SQLite database:
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
    - **Run the ML pipeline** to train classifier and save the model as a pickle file:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. **Run the web app**.

    - Navigate to the `app` directory:
        `cd app`
    - Start the Flask app:
        `python run.py`

4. **View the web app**.
    - Open `http://localhost:5000/` in your browser.

## File structure
    - app
    | - template
    | |- master.html  # main page of web app
    | |- go.html  # classification result page of web app
    |- run.py  # Flask file that runs app (allows user to input new messages and view the classification results)

    - data
    |- categories.csv  # data to process
    |- messages.csv  # data to process
    |- process_data.py
    |- DisasterResponce.db   # database to save clean data to

    - jupyter
    |- ETL Pipeline Preparation.ipynb # cleans and processes message data, storing the cleaned dataset in an SQLite databaset
    |- ML Pipeline Preparation.ipynb # trains a multi-output classifier on the desaster message data, saving the trained model as classifier.pkl

    - models
    |- train_classifier.py
    |- classifier.pkl  # saved model 

    - README.md

    - requirements.txt # list of Python dependencies
    |- To install the necessary dependencies, run: pip install -r requirements.txt