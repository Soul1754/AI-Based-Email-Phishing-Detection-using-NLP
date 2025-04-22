import torch
from lime.lime_text import LimeTextExplainer
from transformers import BertTokenizer
import numpy as np
from data_preprocessing import clean_text
from model import BertLSTMClassifier

class EmailExplainer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertLSTMClassifier.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.explainer = LimeTextExplainer(class_names=['Legitimate', 'Phishing'])

    def predict_proba(self, texts):
        """Prediction probability for LIME."""
        processed_texts = [clean_text(text) for text in texts]
        
        # Tokenize
        encodings = self.tokenizer(
            processed_texts,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()

    def explain_prediction(self, text, num_features=10):
        """Explain why the model made its prediction."""
        exp = self.explainer.explain_instance(
            text, 
            self.predict_proba,
            num_features=num_features,
            num_samples=100
        )
        
        # Get prediction
        probs = self.predict_proba([text])[0]
        prediction = "Phishing" if probs[1] > 0.5 else "Legitimate"
        confidence = probs[1] if prediction == "Phishing" else probs[0]
        
        # Get top contributing words
        word_importance = dict(exp.as_list())
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'word_importance': word_importance,
            'html': exp.as_html()
        }

if __name__ == "__main__":
    # Example usage
    model_path = "/Users/siddhantgaikwad/Developer/College/TY/CS/CP/email_detection/models/bert_email_classifier.pt"
    explainer = EmailExplainer(model_path)
    
    # Example email
    email_text = """
    Dear User, Your account security has been compromised. 
    Click here immediately to verify your identity: http://suspicious-link.com
    Failure to act will result in account termination.
    """
    
    explanation = explainer.explain_prediction(email_text)
    print(f"\nPrediction: {explanation['prediction']}")
    print(f"Confidence: {explanation['confidence']:.2%}")
    print("\nTop contributing words and their importance:")
    for word, importance in explanation['word_importance'].items():
        print(f"{word}: {importance:.4f}")