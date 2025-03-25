import torch
import torch.nn as nn
from transformers import BertModel
from torch.utils.data import Dataset

class EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class BertLSTMClassifier(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, dropout_rate=0.1, num_classes=2):
        super(BertLSTMClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_size = self.bert.config.hidden_size
        
        self.lstm = nn.LSTM(
            input_size=self.bert_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        lstm_outputs, _ = self.lstm(bert_outputs.last_hidden_state)
        last_hidden_state = lstm_outputs[:, -1, :]
        
        dropped = self.dropout(last_hidden_state)
        logits = self.classifier(dropped)
        
        return logits
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
    @classmethod
    def load_model(cls, path, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        return model