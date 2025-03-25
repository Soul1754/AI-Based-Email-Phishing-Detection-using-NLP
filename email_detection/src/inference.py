import torch
from transformers import BertTokenizer
import argparse
import pandas as pd
from model import BertEmailClassifier
from data_preprocessing import clean_text

def load_model_and_tokenizer(model_path):
    """Load the trained model and tokenizer."""
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertEmailClassifier.load_model(model_path)
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def predict_email(text, model, tokenizer, device, max_length=128):
    """
    Predict whether an email is spam or not.
    
    Args:
        text: Email text
        model: Trained model
        tokenizer: BERT tokenizer
        device: Device to run inference on
        max_length: Maximum sequence length
        
    Returns:
        Prediction (0 for ham, 1 for spam) and confidence score
    """
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Tokenize the text
    encoding = tokenizer(
        cleaned_text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence

def batch_predict(texts, model, tokenizer, device, max_length=128):
    """
    Make predictions for a batch of emails.
    
    Args:
        texts: List of email texts
        model: Trained model
        tokenizer: BERT tokenizer
        device: Device to run inference on
        max_length: Maximum sequence length
        
    Returns:
        List of predictions and confidence scores
    """
    # Clean the texts
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Tokenize the texts
    encodings = tokenizer(
        cleaned_texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1).tolist()
        confidences = [probabilities[i][predictions[i]].item() for i in range(len(predictions))]
    
    return predictions, confidences

def main(args):
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args.model_path)
    
    if args.input_file:
        # Load emails from file
        try:
            df = pd.read_csv(args.input_file)
            if 'text' not in df.columns:
                raise ValueError("Input file must contain a 'text' column")
            
            texts = df['text'].tolist()
            predictions, confidences = batch_predict(
                texts, model, tokenizer, device, max_length=args.max_length
            )
            
            # Add predictions to dataframe
            df['prediction'] = predictions
            df['confidence'] = confidences
            df['prediction_label'] = df['prediction'].apply(
                lambda x: "Spam" if x == 1 else "Ham"
            )
            
            # Save results
            output_file = args.output_file or args.input_file.replace('.csv', '_predictions.csv')
            df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")
            
        except Exception as e:
            print(f"Error processing input file: {e}")
    
    elif args.text:
        # Make prediction for a single email
        prediction, confidence = predict_email(
            args.text, model, tokenizer, device, max_length=args.max_length
        )
        
        label = "Spam" if prediction == 1 else "Ham"
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.4f}")
    
    else:
        print("Please provide either an input file or email text")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with trained BERT model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--input_file", type=str, help="Path to CSV file containing emails")
    parser.add_argument("--output_file", type=str, help="Path to save predictions")
    parser.add_argument("--text", type=str, help="Email text to classify")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    
    args = parser.parse_args()
    main(args)