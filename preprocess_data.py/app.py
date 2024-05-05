from flask import Flask, render_template, request
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize

app = Flask(__name__)

model = tf.keras.models.load_model('climate_chatbot_model')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)

def generate_response(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=100, padding='post')
    predictions = model.predict(sequence)
    predicted_index = tf.argmax(predictions, axis=-1).numpy()[0]
    response = tokenizer.sequences_to_texts([predicted_index])[0]
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    response = generate_response(user_input)
    return render_template('index.html', user_input=user_input, response=response)

if __name__ == '__main__':
    app.run(debug=True)
