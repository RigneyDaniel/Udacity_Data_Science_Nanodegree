import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('LabelledMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    news_messages = df[df['genre']=='news'].iloc[:,4:].sum().reset_index().sort_values(0, ascending=False).head(5)
    news_messages[0] = (news_messages[0] / df[df['genre']=='news'].shape[0] * 100)
    news_messages_names = list(news_messages['index'])
    news_messages_counts = news_messages[0]
    
    social_messages = df[df['genre']=='social'].iloc[:,4:].sum().reset_index().sort_values(0, ascending=False).head(5)
    social_messages[0] = (social_messages[0] / df[df['genre']=='social'].shape[0] * 100)
    social_messages_names = list(social_messages['index'])
    social_messages_counts = social_messages[0]
    
    direct_messages = df[df['genre']=='direct'].iloc[:,4:].sum().reset_index().sort_values(0, ascending=False).head(5)
    direct_messages[0] = (direct_messages[0] / df[df['genre']=='direct'].shape[0] * 100)
    direct_messages_names = list(direct_messages['index'])
    direct_messages_counts = direct_messages[0]

    graphs = [
            {
                'data': [
                    Bar(
                        x=genre_names,
                        y=genre_counts
                    )
                ],
    
                'layout': {
                    'title': 'Distribution of Message Genres',
                    'yaxis': {
                        'title': "Count"
                    },
                    'xaxis': {
                        'title': "Genre"
                    }
                }
            },
            {
                'data': [
                    Bar(
                        x=news_messages_names,
                        y=news_messages_counts
                    )
                ],
    
                'layout': {
                    'title': 'Percent of News Messages by Category',
                    'yaxis': {
                        'title': "Percent"
                    },
                    'xaxis': {
                        'title': "Category"
                    }
                }
            },
            {
                'data': [
                    Bar(
                        x=social_messages_names,
                        y=social_messages_counts
                    )
                ],
    
                'layout': {
                    'title': 'Percent of Social Messages by Category',
                    'yaxis': {
                        'title': "Percent"
                    },
                    'xaxis': {
                        'title': "Category"
                    }
                }
            },
            {
                'data': [
                    Bar(
                        x=direct_messages_names,
                        y=direct_messages_counts
                    )
                ],
    
                'layout': {
                    'title': 'Percent of Direct Messages by Category',
                    'yaxis': {
                        'title': "Percent"
                    },
                    'xaxis': {
                        'title': "Category"
                    }
                }
            }    
        ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()