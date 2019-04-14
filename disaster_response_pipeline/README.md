# Disaster Response Pipleline Project**
This project was completed to as the 5th assessment of the Udactiy Data Science Nanodegree.

----

## Project Overview
Using disaster message data provided by Figure Eight, this project builds a classification model that categorises new messages so that new messages can be sent to the correct disaster response organisation.

Messages are entered into the web application and the categories are displayed visually.

---

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

---

## Files

* process_data.py - loads raw message and category data, combines data, cleans data and saves as an SQLite database.
* train_classifier.py - tokenises messages, constructs a multi-output, random forest classifier model using cross validation. Trains the model on labelled messages and evaulates the models performance using an out-of-sample set of messages.
* run.py - builds a web application and displays visualisations.

---

## Libraries
Python: pandas, sklearn, nltk, sqlalchmy, flask, json, plotly
CSS: Bootstrap