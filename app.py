import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Define the path to the unzipped directory
model_path = 'saved_model'

# Load the tokenizer and model from the directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define label mappings
label2id = {'negative': 0, 'positive': 1}
id2label = {value: key for key, value in label2id.items()}

# Prediction function
def predict_sentiment(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        sentiments = [id2label[pred.item()] for pred in predictions]
    return sentiments

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    texts = data.get("texts")
    sentiments = predict_sentiment(texts)
    return jsonify(sentiments=sentiments)

if __name__ == '__main__':
    app.run(debug=True)
