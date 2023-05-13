import json
import nltk
import random
import string
import pickle
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Package sentence tokenizer
nltk.download('punkt')
# Package lemmatization
nltk.download('wordnet')
# Package multilingual wordnet data
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
le = LabelEncoder()

from keras.models import load_model
model = load_model('model-complete/model_chatbot.h5')
intents = json.loads(open('kampus_merdeka_cmplt.json').read())
words = pickle.load(open('model-complete/words.pkl','rb'))
classes = pickle.load(open('model-complete/classes.pkl','rb'))
tokenizer = pickle.load(open('model-complete/tokenizers.pkl','rb'))
input_shape = 16

le.fit(classes)

def preprocessing(sentences):
    texts_p = []
    print(sentences)
    prediction_input = [letters.lower() for letters in sentences if letters not in string.punctuation]
    print(prediction_input)
    prediction_input = ''.join(prediction_input)
    print(prediction_input)
    texts_p.append(prediction_input)
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    print(prediction_input)
    prediction_input = np.array(prediction_input).reshape(-1)
    print(prediction_input)
    prediction_input = pad_sequences([prediction_input],input_shape)
    print(prediction_input)
    return prediction_input

def predict_class(sentence, model):
    p = preprocessing(sentence)
    print(p)
    output = model.predict(p)
    print(output)
    output = output.argmax()
    print(output)
    return output

def getResponse(output, intents_json):
    tag = le.inverse_transform([output])[0]
    print(tag)
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            print(result)
            break
        else:
            result = "Maaf chatbot tidak mengerti, coba tanyakan lagi"
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    print(ints)
    res = getResponse(ints, intents)
    print(res)
    return res


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print(userText)
    return chatbot_response(userText)

if __name__ == "__main__":
    app.run()