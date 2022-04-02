from flask import Flask, jsonify, request

import joblib

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

app = Flask(__name__)

def preprocess(data):
  data = [x.lower() for x in data.split()]
  data0 = ' '.join(data)
  all_list = [char for char in data0 if char not in string.punctuation]
  data1 = ''.join(all_list)
  stop = stopwords.words('english')
  data = ' '.join([word for word in data1.split() if word not in (stop)])

  return data

@app.route('/api/predict', methods=['GET'])
def home():
    modelUsed = ''
    if(request.method == 'GET'):
        vectorizer = joblib.load('models/vectorizer.pkl')
        if (request.json['model'] == 'DT'):
            model = joblib.load('models/DecisionTree_model.pkl')
            modelUsed = 'DecisionTree'
        elif (request.json['model'] == 'LR'):
            model = joblib.load('models/LogisticRegression_model.pkl')
            modelUsed = 'Logistic Regression'
        elif (request.json['model'] == 'BY'):
            model = joblib.load('models/bayes_model.pkl')
            modelUsed = 'Bayes'
        elif (request.json['model'] == 'RF'):
            model = joblib.load('models/RandomForest_model.pkl')
            modelUsed = 'Random Forest'
        else:
            model = joblib.load('models/DecisionTree_model.pkl')
            modelUsed = 'DecisionTree'

        newsBody = vectorizer.transform([preprocess(request.json['newsBody'])])
        pred = model.predict(newsBody)

        return jsonify({'model_used': modelUsed,'prediction': pred[0]})


if __name__ == '__main__':
    app.run(debug=True)
