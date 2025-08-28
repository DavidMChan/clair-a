from typing import Dict, List

import torch


def infer_preprocess(tokenizer, texts: List[str], max_len: int = 64) -> Dict[str, torch.Tensor]:
    """
    Preprocess texts for inference with the BERT classifier.
    
    Args:
        tokenizer: HuggingFace tokenizer
        texts: List of text strings to preprocess
        max_len: Maximum sequence length
        
    Returns:
        Dictionary containing tokenized inputs ready for the model
    """
    # Tokenize the texts
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    
    return encoded
