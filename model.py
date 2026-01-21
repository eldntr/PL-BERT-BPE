import torch
import torch.nn as nn
from transformers import AlbertConfig, AlbertModel

class MultiTaskModel(nn.Module):
    def __init__(
        self,
        phoneme_vocab_size,
        bpe_vocab_size,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
    ):
        super().__init__()

        config = AlbertConfig(
            vocab_size=phoneme_vocab_size,
            embedding_size=hidden_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
        )

        self.encoder = AlbertModel(config)

        # MLM head: predict phoneme
        self.mlm_head = nn.Linear(hidden_size, phoneme_vocab_size)

        # CTC head: predict BPE (with extra blank token at index 0)
        self.ctc_output_dim = bpe_vocab_size + 1  # +1 for blank
        self.ctc_head = nn.Linear(hidden_size, self.ctc_output_dim)

    def forward(self, input_ids, attention_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, T, H]

        mlm_logits = self.mlm_head(hidden)   # [B, T, phoneme_vocab]
        ctc_logits = self.ctc_head(hidden)   # [B, T, ctc_output_dim]

        return mlm_logits, ctc_logits