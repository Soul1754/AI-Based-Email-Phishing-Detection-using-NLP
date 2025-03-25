import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertModel, BertConfig

class EmailDataset(Dataset):
    """Dataset class for email classification."""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.labels)

class BertEmailClassifier(nn.Module):
    """BERT-based model for email classification."""
    
    def __init__(self, num_classes=2, dropout_rate=0.1):
        super(BertEmailClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get the [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def save_model(self, path):
        """Save the model to the specified path."""
        torch.save(self.state_dict(), path)
        
    @classmethod
    def load_model(cls, path, num_classes=2, dropout_rate=0.1):
        """Load the model from the specified path."""
        model = cls(num_classes=num_classes, dropout_rate=dropout_rate)
        model.load_state_dict(torch.load(path))
        return model