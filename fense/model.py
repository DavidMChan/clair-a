from torch import nn
from transformers import AutoModel


class BERTFlatClassifier(nn.Module):
    def __init__(self, model_type='distilbert-base-uncased', num_classes=2, dropout=0.1):
        super(BERTFlatClassifier, self).__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        
        self.encoder = AutoModel.from_pretrained(model_type)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use pooled output (CLS token representation)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def load_state_dict(self, state_dict, strict=True):
        # Filter out the problematic position_ids key
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if 'position_ids' not in key:
                filtered_state_dict[key] = value
        
        # Load with strict=False to handle any remaining compatibility issues
        return super().load_state_dict(filtered_state_dict, strict=False)
