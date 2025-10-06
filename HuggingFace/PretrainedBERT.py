import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, AdamW

class PretrainedBERT(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, output_nodes, freeze_bert=False):
        """
        @param    output_nodes: Number of output nodes for the classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(PretrainedBERT, self).__init__()
        
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 512, 64, output_nodes
        
        # Instantiate BERT model
        from transformers import BertConfig
        config = BertConfig(
            max_position_embeddings=255,
            hidden_size=512,
            num_attention_heads=16,
            num_hidden_layers=2,
            type_vocab_size=1,
            intermediate_size=1024,
            vocab_size=16
        )
        
        self.bert = BertModel(config)
        
        # Instantiate a one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out),
        )

        # Freeze the BERT model if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size, max_length)
        @param    attention_mask (torch.Tensor): a tensor that holds attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size, num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

def initialize_pretrained_bert(output_nodes, epochs=4):
    """Initialize the Pretrained BERT and the optimizer."""
    # Instantiate Pretrained BERT
    pretrained_bert = PretrainedBERT(output_nodes, freeze_bert=False)

    # Create the optimizer
    optimizer = AdamW(pretrained_bert.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )
    
    return pretrained_bert, optimizer
