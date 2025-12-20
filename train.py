import os
import shutil
import os.path as osp
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch import nn

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from transformers import AlbertConfig, AlbertModel
from datasets import load_from_disk, DatasetDict

import yaml
import editdistance

from model import MultiTaskModel
from text_tokenizer import TextTokenizer
from dataloader_ctc import build_dataloader


# ============================================================
# DATASET SPLIT (DETERMINISTIK)
# ============================================================

def split_and_save_dataset(data_dir, out_dir, seed=1234):
    ds = load_from_disk(data_dir)

    split1 = ds.train_test_split(test_size=0.1, seed=seed)
    split2 = split1["test"].train_test_split(test_size=0.5, seed=seed)

    dsd = DatasetDict({
        "train": split1["train"],
        "val": split2["train"],
        "test": split2["test"],
    })

    dsd.save_to_disk(out_dir)
    return dsd


# ============================================================
# METRICS
# ============================================================

def normalize_ctc_seq(seq, blank=0):
    """
    Normalisasi CTC sequence:
    - remove blank
    - collapse repeat
    - unshift token (id-1)
    """
    out = []
    prev = blank
    for x in seq:
        if x == blank:
            prev = x
            continue
        if x == prev:
            prev = x
            continue
        out.append(x - 1)   # UN-SHIFT
        prev = x
    return out


def unshift_targets_only(seq):
    """
    Target CTC TIDAK boleh collapse repeat.
    Target juga tidak punya blank.
    Jadi cukup UN-SHIFT (id-1).
    """
    return [x - 1 for x in seq]


def token_error_rate(preds, targets):
    """
    Token Error Rate (edit distance / target length)
    """
    total = 0
    dist = 0
    for p, t in zip(preds, targets):
        if len(t) == 0:
            continue
        dist += editdistance.eval(p, t)
        total += len(t)
    return dist / max(1, total)


def f1_presence_based(preds, targets):
    """
    Presence-based F1 (set overlap).
    Lebih stabil untuk CTC di early training.
    """
    f1_vals = []

    for p, t in zip(preds, targets):
        if len(t) == 0:
            continue

        p_set = set(p)
        t_set = set(t)

        tp = len(p_set & t_set)
        fp = len(p_set - t_set)
        fn = len(t_set - p_set)

        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)

        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)

        f1_vals.append(f1)

    return sum(f1_vals) / max(1, len(f1_vals))


def mlm_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == labels[mask]).float().mean().item()


def ctc_greedy_decode(logits, input_lengths, blank=0):
    preds = logits.argmax(dim=-1)  # [B, T]
    results = []

    for seq, L in zip(preds, input_lengths):
        out = []
        prev = blank
        for t in seq[:L]:
            t = t.item()
            if t != blank and t != prev:
                out.append(t)
            prev = t
        results.append(out)

    return results


# ============================================================
# EVALUATION (LOGITS-BASED, BENAR)
# ============================================================

@torch.no_grad()
def evaluate(model, val_loader, accelerator):
    model.eval()

    tot_loss = tot_mlm = tot_ctc = 0.0
    tot_acc = 0.0
    n = 0

    all_preds = []
    all_tgts = []

    # DEBUG
    empty_pred = 0
    nonempty_tgt = 0

    for batch in val_loader:
        # ----- forward -----
        loss, logs = model.compute_loss(batch)

        mlm_logits, ctc_logits = model(
            batch["phoneme_input"],
            batch["attention_mask"]
        )

        # ----- MLM accuracy -----
        acc = mlm_accuracy(mlm_logits, batch["mlm_labels"])

        # ----- CTC greedy decode (SHIFTED SPACE) -----
        preds_shifted = ctc_greedy_decode(
            ctc_logits,
            batch["input_lengths"],
            blank=0
        )

        # ----- reconstruct SHIFTED targets -----
        targets_shifted = []
        idx = 0
        for L in batch["target_lengths"].tolist():
            targets_shifted.append(
                batch["ctc_targets"][idx:idx + L].tolist()
            )
            idx += L

        # ----- NORMALIZE -----
        # preds: CTC normalize (remove blank + collapse + unshift)
        preds = [normalize_ctc_seq(p, blank=0) for p in preds_shifted]
        
        # tgts: JANGAN collapse, cukup unshift
        tgts  = [unshift_targets_only(t) for t in targets_shifted]

        # ----- DEBUG COUNTS -----
        for p, t in zip(preds, tgts):
            if len(t) > 0:
                nonempty_tgt += 1
            if len(p) == 0:
                empty_pred += 1

        all_preds.extend(preds)
        all_tgts.extend(tgts)

        # ----- aggregate losses -----
        tot_loss += float(logs["loss_total"])
        tot_mlm  += float(logs["loss_mlm"])
        tot_ctc  += float(logs["loss_ctc"])
        tot_acc  += float(acc)
        n += 1

    # ===== FINAL METRICS =====
    ter = token_error_rate(all_preds, all_tgts)
    f1  = f1_presence_based(all_preds, all_tgts)

    model.train()

    # DEBUG PRINT (sekali per eval)
    if accelerator.is_main_process:
        accelerator.print(
            f"[VAL DEBUG] empty_pred={empty_pred}/{len(all_preds)} "
            f"(nonempty_tgt={nonempty_tgt})"
        )

    return {
        "loss": tot_loss / max(1, n),
        "mlm_loss": tot_mlm / max(1, n),
        "ctc_loss": tot_ctc / max(1, n),
        "mlm_acc": tot_acc / max(1, n),
        "wer": ter,   # sebenarnya Token Error Rate (ID-based)
        "f1": f1,     # presence-based F1
    }


# ============================================================
# TRAINING
# ============================================================

def train():
    config = yaml.safe_load(open("Configs/config.yml"))

    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "no"),
        split_batches=True,
        kwargs_handlers=[ddp],
    )

    # ---------- DATA ----------
    split_dir = config["data_folder"] + "_split"
    if not os.path.exists(split_dir):
        split_and_save_dataset(config["data_folder"], split_dir)

    dataset = load_from_disk(split_dir)
    train_ds, val_ds = dataset["train"], dataset["val"]

    train_loader, train_wrap = build_dataloader(
        train_ds,
        batch_size=config["batch_size"],
        mlm_prob=config.get("mlm_prob", 0.15),
        shuffle_batches=True,
    )

    # 🔥 MLM AKTIF JUGA DI VAL
    val_loader, _ = build_dataloader(
        val_ds,
        batch_size=config["batch_size"],
        mlm_prob=config.get("mlm_prob", 0.15),
        shuffle_batches=False,
    )

    # ---------- MODEL ----------
    num_tokens = len(train_wrap.text_cleaner.word_index_dictionary)

    text_tok = TextTokenizer(config["dataset_params"]["tokenizer"])
    num_vocab = text_tok.vocab_size + 1

    enc_cfg = AlbertConfig(**config["model_params"])
    encoder = AlbertModel(enc_cfg)

    model = MultiTaskModel(
        encoder=encoder,
        num_tokens=num_tokens,
        num_vocab=num_vocab,
        hidden_size=config["model_params"]["hidden_size"],
        ctc_blank_id=0,
    )

    optimizer = AdamW(model.parameters(), lr=config.get("lr", 1e-4))

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # ---------- TRAIN LOOP ----------
    accum_steps = config.get("grad_accum", 8)
    log_interval = config["log_interval"]
    save_interval = config["save_interval"]
    num_steps = config["num_steps"]

    step = 0
    optimizer.zero_grad()

    accelerator.print("Start training...")
    accelerator.print(f"Gradient accumulation steps: {accum_steps}")

    for batch in train_loader:
        step += 1

        loss, logs = model.compute_loss(batch)
        loss = loss / accum_steps
        accelerator.backward(loss)

        if step % accum_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if step % log_interval == 0:
            accelerator.print(
                f"[TRAIN {step}/{num_steps}] "
                f"Total: {logs['loss_total']:.4f} | "
                f"MLM: {logs['loss_mlm']:.4f} | "
                f"CTC: {logs['loss_ctc']:.4f} | (accum {accum_steps}x)"
            )

            metrics = evaluate(model, val_loader, accelerator)
            accelerator.print(
                f"[VAL] "
                f"Loss {metrics['loss']:.4f} | "
                f"MLM {metrics['mlm_loss']:.4f} | "
                f"CTC {metrics['ctc_loss']:.4f} | "
                f"Acc {metrics['mlm_acc']:.3f} | "
                f"WER {metrics['wer']:.3f} | "
                f"F1 {metrics['f1']:.3f}"
            )

        if step % save_interval == 0:
            accelerator.save(
                {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                },
                osp.join(config["log_dir"], f"step_{step}.t7")
            )

        if step >= num_steps:
            break


if __name__ == "__main__":
    from accelerate import notebook_launcher
    notebook_launcher(train, args=(), num_processes=1)
