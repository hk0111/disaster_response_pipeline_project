# Disaster Response Pipeline Project
Building a machine learning model to classify disaster messages so they can be sent to an appropriate disaster relief agency.

### Table of Contents
1. [Installation](#installation)
2. [File Description](#description)
3. [Instructions](#insutrctions)
4. [Licensing, Authors, Acknowledgements](#licensing)

## Installation <a name="installation"></a>
To run the code here, you will need libraries specified in [`requirements.txt`](requirements.txt) and nltk corpora listed in [`nltk.txt`](nltk.txt). The code should run with no issues on Python 3.x.

## File Description <a name="description"></a>
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
| - run.py  # back-end for web app

- data
| - disaster_categories.csv  # data to process 
| - disaster_messages.csv  # data to process
| - process_data.py # script to process data
| - DisasterResponse.db   # database to save the clean data

- models
| - train_classifier.py # script to train ML model
| - classifier.pkl  # saved model 
| - utils # contains tokenizer function

- utils # contains tokenizer function

- my_app.py # Runs the web app

- requirements.txt # packages required

- nltk.txt # NLTK corpora required

- Procfile # To run the web app on heroku
```
## Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves the model
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the project's root directory to run your web app.

    `python my_app.py`

3. Go to http://0.0.0.0:3001/ to view the web app.

4. The web app is deployed to heroku that can be accessed by clicking this [link](https://classify-disaster-message-hk.herokuapp.com/).

## Licensing, Authors, Acknowledgements <a name="licensing"></a>
The datasets here are provided by [Udacity](https://www.udacity.com/) and [Figure Eight](https://appen.com/). Terms and conditions for using the datasets can be found on the website.