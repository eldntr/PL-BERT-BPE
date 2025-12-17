import torch
from torch import nn
import torch.nn.functional as F


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        encoder,
        num_tokens,        # vocab IPA (TextCleaner)
        num_vocab,         # vocab BPE + 1 (blank)
        hidden_size,
        ctc_blank_id=0,
    ):
        super().__init__()

        self.encoder = encoder

        self.mlm_head = nn.Linear(hidden_size, num_tokens)
        self.ctc_head = nn.Linear(hidden_size, num_vocab)

        self.ctc_loss_fn = nn.CTCLoss(
            blank=ctc_blank_id,
            zero_infinity=True
        )

    def forward(self, phoneme_input, attention_mask=None):
        out = self.encoder(
            input_ids=phoneme_input,
            attention_mask=attention_mask
        )
        h = out.last_hidden_state                     # [B,T,H]
        mlm_logits = self.mlm_head(h)                 # [B,T,num_tokens]
        ctc_logits = self.ctc_head(h)                 # [B,T,num_vocab]
        return mlm_logits, ctc_logits

    def compute_loss(self, batch, mlm_weight=1.0, ctc_weight=1.0):
        phoneme_input  = batch["phoneme_input"]
        attention_mask = batch["attention_mask"]
        mlm_labels     = batch["mlm_labels"]
        ctc_targets    = batch["ctc_targets"]
        input_lengths  = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        mlm_logits, ctc_logits = self.forward(
            phoneme_input,
            attention_mask
        )

        # ===== MLM loss =====
        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            mlm_labels.view(-1),
            ignore_index=-100
        )

        # ===== CTC loss =====
        log_probs = F.log_softmax(ctc_logits, dim=-1)   # [B,T,V]
        log_probs = log_probs.transpose(0, 1)          # [T,B,V]

        ctc_loss = self.ctc_loss_fn(
            log_probs,
            ctc_targets,
            input_lengths,
            target_lengths
        )

        total_loss = mlm_weight * mlm_loss + ctc_weight * ctc_loss

        return total_loss, {
            "loss_total": total_loss.detach(),
            "loss_mlm": mlm_loss.detach(),
            "loss_ctc": ctc_loss.detach(),
        }
