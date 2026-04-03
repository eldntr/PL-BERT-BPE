import torch
import torch.nn as nn
from transformers import AlbertConfig, AlbertModel

class MultiTaskModel(nn.Module):
    def __init__(
        self,
        encoder,
        num_vocab,
        num_tokens,
        hidden_size,
    ):
        super().__init__()

        self.encoder = encoder

        # MLM head: predict phoneme
        self.mlm_head = nn.Linear(hidden_size, num_tokens)

        # BPE head: predict BPE
        self.ctc_head = nn.Linear(hidden_size, num_vocab)

    def forward(self, input_ids, attention_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, T, H]

        mlm_logits = self.mlm_head(hidden)   # [B, T, phoneme_vocab]
        ctc_logits = self.ctc_head(hidden)   # [B, T, ctc_output_dim]

        return mlm_logits, ctc_logits