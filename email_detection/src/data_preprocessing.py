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
    Load and preprocess phishing email data.
    """
    # Load the dataset
    try:
        df = pd.read_csv(data_path)
        # Print column names to debug
        print("Available columns:", df.columns.tolist())
    except:
        raise ValueError("Unable to load CSV file. Please check the file path and format.")
    
    # Rename columns if needed (adjust these based on your CSV structure)
    column_mapping = {
        'text': 'text',  # Update this with actual column name
        'label': 'label'  # Update this with actual column name
    }
    
    # Only rename columns if they exist and need renaming
    existing_columns = df.columns.tolist()
    column_mapping = {k: v for k, v in column_mapping.items() if k in existing_columns}
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Convert label to numeric if needed (0 for legitimate, 1 for phishing)
    if 'label' in df.columns and df['label'].dtype == 'object':
        df['label'] = (df['label'].str.lower() == 'phishing').astype(int)
    
    # Clean the text data
    text_column = 'text' if 'text' in df.columns else df.columns[0]  # Use first column if 'text' not found
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Drop rows with empty text after cleaning
    df = df.dropna(subset=['cleaned_text'])
    
    # Split the data into train and test sets
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    print(f"Total dataset size: {len(df)}")
    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")
    print(f"Phishing emails: {df['label'].sum()}")
    print(f"Legitimate emails: {len(df) - df['label'].sum()}")
    
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
    data_path = "/Users/siddhantgaikwad/Developer/College/TY/CS/CP/email_detection/data/phishing_email.csv"
    train_encodings, test_encodings, train_labels, test_labels, tokenizer = prepare_data(data_path)
    print("Data preprocessing completed successfully!")