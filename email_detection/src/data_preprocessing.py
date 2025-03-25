import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def clean_text(text):
    """Clean and preprocess text data."""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def load_and_preprocess_data(data_path, test_size=0.2, random_state=42):
    """
    Load and preprocess email data.
    
    Args:
        data_path: Path to the dataset
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Preprocessed train and test datasets
    """
    # Load the dataset
    try:
        df = pd.read_csv(data_path)
    except:
        try:
            df = pd.read_excel(data_path)
        except:
            raise ValueError("Unsupported file format. Please provide CSV or Excel file.")
    
    # Check if the dataset has the expected columns
    required_columns = ['text', 'label']
    if not all(col in df.columns for col in required_columns):
        # Try to infer columns
        if len(df.columns) >= 2:
            # Assume first column is text and second is label
            df.columns = ['text', 'label'] + list(df.columns[2:])
        else:
            raise ValueError(f"Dataset must contain columns: {required_columns}")
    
    # Clean the text data
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Drop rows with empty text after cleaning
    df = df.dropna(subset=['cleaned_text'])
    
    # Split the data into train and test sets
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")
    
    return train_df, test_df

def tokenize_data(train_df, test_df, tokenizer, max_length=128):
    """
    Tokenize the text data using BERT tokenizer.
    
    Args:
        train_df: Training dataframe
        test_df: Testing dataframe
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tokenized train and test datasets
    """
    # Tokenize training data
    train_encodings = tokenizer(
        train_df['cleaned_text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Tokenize testing data
    test_encodings = tokenizer(
        test_df['cleaned_text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Convert labels to tensors
    train_labels = train_df['label'].values
    test_labels = test_df['label'].values
    
    return train_encodings, test_encodings, train_labels, test_labels

def prepare_data(data_path, max_length=128):
    """
    Complete data preparation pipeline.
    
    Args:
        data_path: Path to the dataset
        max_length: Maximum sequence length for tokenization
        
    Returns:
        Processed data ready for model training
    """
    # Load and preprocess data
    train_df, test_df = load_and_preprocess_data(data_path)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize data
    train_encodings, test_encodings, train_labels, test_labels = tokenize_data(
        train_df, test_df, tokenizer, max_length
    )
    
    return train_encodings, test_encodings, train_labels, test_labels, tokenizer

if __name__ == "__main__":
    # Example usage
    data_path = "/Users/siddhantgaikwad/Developer/College/TY/CS/CP/email_detection/data/email_dataset.csv"
    train_encodings, test_encodings, train_labels, test_labels, tokenizer = prepare_data(data_path)
    print("Data preprocessing completed successfully!")