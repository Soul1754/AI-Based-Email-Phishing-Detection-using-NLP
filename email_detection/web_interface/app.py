from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer
import torch
from model import BertLSTMClassifier
from data_preprocessing import clean_text
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)

# Load model and tokenizer
model_path = 'models/trained_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertLSTMClassifier.load_model(model_path)
model.to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
explainer = LimeTextExplainer(class_names=['Legitimate', 'Phishing'])

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    cleaned_text = clean_text(text)
    encoding = tokenizer(
        cleaned_text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

    exp = explainer.explain_instance(
        text, 
        lambda x: model(input_ids=tokenizer(x, return_tensors='pt')['input_ids'].to(device), attention_mask=tokenizer(x, return_tensors='pt')['attention_mask'].to(device)).detach().cpu().numpy(),
        num_features=10,
        num_samples=100
    )

    word_importance = dict(exp.as_list())

    return jsonify({
        'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
        'confidence': confidence,
        'word_importance': word_importance
    })

if __name__ == '__main__':
    app.run(debug=True)