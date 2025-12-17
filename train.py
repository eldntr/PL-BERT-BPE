import os
import shutil
import os.path as osp
from datetime import datetime

import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from transformers import AlbertConfig, AlbertModel
from torch.optim import AdamW
from datasets import load_from_disk, DatasetDict
import yaml
import editdistance
from sklearn.metrics import f1_score

from model import MultiTaskModel
from text_tokenizer import TextTokenizer
from dataloader_ctc import FilePathDataset, LengthBucketSampler, collate_fn, build_dataloader


def split_and_save_dataset(data_dir, out_dir, seed=42):
    """Split dataset deterministik: 90% train, 5% val, 5% test"""
    ds = load_from_disk(data_dir)

    # 90% train, 10% temp
    split1 = ds.train_test_split(test_size=0.1, seed=seed)

    # dari 10% → 5% val, 5% test
    split2 = split1["test"].train_test_split(test_size=0.5, seed=seed)

    dataset_dict = DatasetDict({
        "train": split1["train"],
        "val": split2["train"],
        "test": split2["test"],
    })

    dataset_dict.save_to_disk(out_dir)
    print("Split dataset saved:", dataset_dict)
    return dataset_dict


def mlm_accuracy(logits, labels):
    """
    MLM Accuracy.
    logits: [B,T,V]
    labels: [B,T] with -100 ignored
    """
    preds = logits.argmax(dim=-1)
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    correct = (preds[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item()


def ctc_greedy_decode(logits, input_lengths, blank=0):
    """
    CTC Greedy Decode.
    logits: [B,T,V]
    return: List[List[int]]
    """
    preds = logits.argmax(dim=-1)  # [B,T]
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


def wer(preds, targets):
    """
    Word Error Rate (WER).
    preds, targets: List[List[int]]
    """
    total_words = 0
    total_err = 0
    for p, t in zip(preds, targets):
        total_err += editdistance.eval(p, t)
        total_words += len(t)
    return total_err / max(1, total_words)


def token_f1(preds, targets):
    """
    Token-level F1.
    preds, targets: List[List[int]]
    """
    flat_p = []
    flat_t = []
    for p, t in zip(preds, targets):
        L = min(len(p), len(t))
        flat_p.extend(p[:L])
        flat_t.extend(t[:L])
    if len(flat_t) == 0:
        return 0.0
    return f1_score(flat_t, flat_p, average="micro")


@torch.no_grad()
def evaluate(model, val_loader, accelerator):
    """Evaluation loop."""
    model.eval()

    tot_loss = tot_mlm = tot_ctc = 0
    tot_acc = 0
    all_preds = []
    all_tgts = []
    n = 0

    for batch in val_loader:
        loss, logs = model.compute_loss(batch)
        mlm_logits, ctc_logits = model(
            batch["phoneme_input"],
            batch["attention_mask"]
        )

        acc = mlm_accuracy(mlm_logits, batch["mlm_labels"])

        preds = ctc_greedy_decode(
            ctc_logits,
            batch["input_lengths"]
        )

        # reconstruct target list
        targets = []
        idx = 0
        for L in batch["target_lengths"]:
            targets.append(batch["ctc_targets"][idx:idx+L].tolist())
            idx += L

        tot_loss += logs["loss_total"]
        tot_mlm  += logs["loss_mlm"]
        tot_ctc  += logs["loss_ctc"]
        tot_acc  += acc
        all_preds.extend(preds)
        all_tgts.extend(targets)
        n += 1

    model.train()

    return {
        "loss": tot_loss / n,
        "mlm_loss": tot_mlm / n,
        "ctc_loss": tot_ctc / n,
        "mlm_acc": tot_acc / n,
        "wer": wer(all_preds, all_tgts),
        "f1": token_f1(all_preds, all_tgts),
    }


def train():
    config_path = "Configs/config.yml"
    config = yaml.safe_load(open(config_path))

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "no"),
        split_batches=True,
        kwargs_handlers=[ddp_kwargs],
    )

    # Split dataset jika belum ada
    split_dir = config["data_folder"] + "_split"
    if not os.path.exists(split_dir):
        accelerator.print("Splitting dataset...")
        split_and_save_dataset(
            data_dir=config["data_folder"],
            out_dir=split_dir,
            seed=1234
        )
    
    dataset = load_from_disk(split_dir)
    train_ds = dataset["train"]
    val_ds = dataset["val"]

    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    
    # Setup log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_log_file = osp.join(log_dir, f"train_metrics_{timestamp}.txt")
    val_log_file = osp.join(log_dir, f"val_metrics_{timestamp}.txt")
    
    # Write headers
    with open(train_log_file, "w") as f:
        f.write("step,loss_total,loss_mlm,loss_ctc\n")
    with open(val_log_file, "w") as f:
        f.write("step,loss,mlm_loss,ctc_loss,mlm_acc,wer,f1\n")

    batch_size = config["batch_size"]
    mlm_prob = config.get("mlm_prob", 0.15)

    train_loader, train_wrap = build_dataloader(
        train_ds,
        batch_size=batch_size,
        num_workers=config.get("num_workers", 0),
        mlm_prob=mlm_prob,
        shuffle_batches=True,
    )
    
    val_loader, val_wrap = build_dataloader(
        val_ds,
        batch_size=batch_size,
        num_workers=config.get("num_workers", 0),
        mlm_prob=0.0,  # NO masking saat eval
        shuffle_batches=False,
    )

    num_tokens = len(train_wrap.text_cleaner.word_index_dictionary)

    text_tok = TextTokenizer(model_name=config["dataset_params"]["tokenizer"], use_fast=True)
    num_vocab = text_tok.vocab_size + 1  

    enc_cfg = AlbertConfig(**config["model_params"])
    encoder = AlbertModel(enc_cfg)

    hidden_size = config["model_params"]["hidden_size"]

    model = MultiTaskModel(
        encoder=encoder,
        num_tokens=num_tokens,
        num_vocab=num_vocab,
        hidden_size=hidden_size,
        ctc_blank_id=0,
    )

    iters = 0
    load = False
    try:
        ckpts = [f for f in os.listdir(log_dir) if f.startswith("step_") and f.endswith(".t7")]
        if len(ckpts) > 0:
            iters = sorted([int(f.split("_")[-1].split(".")[0]) for f in ckpts])[-1]
            load = True
    except:
        pass

    optimizer = AdamW(model.parameters(), lr=config.get("lr", 1e-4))

    if load:
        ckpt_path = osp.join(log_dir, f"step_{iters}.t7")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        state_dict = checkpoint["net"]
        new_state = {}
        for k, v in state_dict.items():
            new_state[k[7:]] = v if k.startswith("module.") else v
        model.load_state_dict(new_state, strict=False)

        optimizer.load_state_dict(checkpoint["optimizer"])
        accelerator.print(f"Checkpoint loaded: {ckpt_path}")

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    num_steps = config["num_steps"]
    log_interval = config["log_interval"]
    save_interval = config["save_interval"]

    accelerator.print("Start training...")
    curr_steps = iters

    for step, batch in enumerate(train_loader):
        curr_steps += 1

        loss, logs = model.compute_loss(batch, mlm_weight=1.0, ctc_weight=1.0)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        if curr_steps % log_interval == 0:
            accelerator.print(
                f"[TRAIN {curr_steps}/{num_steps}] "
                f"Total: {logs['loss_total']:.4f} | "
                f"MLM: {logs['loss_mlm']:.4f} | "
                f"CTC: {logs['loss_ctc']:.4f}"
            )
            
            # Save train metrics to file
            if accelerator.is_main_process:
                with open(train_log_file, "a") as f:
                    f.write(f"{curr_steps},{logs['loss_total']:.6f},{logs['loss_mlm']:.6f},{logs['loss_ctc']:.6f}\n")
            
            # Validation
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
            
            # Save val metrics to file
            if accelerator.is_main_process:
                with open(val_log_file, "a") as f:
                    f.write(
                        f"{curr_steps},{metrics['loss']:.6f},{metrics['mlm_loss']:.6f},"
                        f"{metrics['ctc_loss']:.6f},{metrics['mlm_acc']:.6f},"
                        f"{metrics['wer']:.6f},{metrics['f1']:.6f}\n"
                    )

        if curr_steps % save_interval == 0:
            accelerator.print("Saving checkpoint...")
            accelerator.save(
                {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": curr_steps,
                    "num_tokens": num_tokens,
                    "num_vocab": num_vocab,
                },
                osp.join(log_dir, f"step_{curr_steps}.t7"),
            )

        if curr_steps >= num_steps:
            break


if __name__ == "__main__":
    from accelerate import notebook_launcher
    while True:
        notebook_launcher(train, args=(), num_processes=3, use_port=33389)