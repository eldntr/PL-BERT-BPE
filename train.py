import os
import shutil
import os.path as osp
import pickle
import yaml

import torch
from torch import nn
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs

from torch.optim import AdamW
from transformers import AlbertConfig, AlbertModel
from transformers import AutoTokenizer

from model import MultiTaskModel
from dataloader import build_dataloader
from utils import length_to_mask, scan_checkpoint

from datasets import load_from_disk

import pickle

config_path = "Configs/config.yml" 
config = yaml.safe_load(open(config_path))

with open(config['dataset_params']['token_maps'], 'rb') as handle:
    token_maps = pickle.load(handle)

tokenizer = AutoTokenizer.from_pretrained(config['dataset_params']['tokenizer'])
criterion = nn.CrossEntropyLoss()

num_steps = config['num_steps']
log_interval = config['log_interval']
save_interval = config['save_interval']

def train():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    curr_steps = 0
    
    dataset = load_from_disk(config["data_folder"])
    if "train" in dataset:
        dataset = dataset["train"]

    log_dir = config['log_dir']
    if not osp.exists(log_dir): 
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    
    batch_size = config["batch_size"]
    train_loader = build_dataloader(dataset, 
                                    batch_size=batch_size, 
                                    num_workers=4, 
                                    dataset_config=config['dataset_params'])

    albert_base_configuration = AlbertConfig(**config['model_params'])
    
    bert_encoder = AlbertModel(albert_base_configuration)
    bert = MultiTaskModel(bert_encoder, 
                          num_vocab=1 + max([m['token'] for m in token_maps.values()]), 
                          num_tokens=config['model_params']['vocab_size'],
                          hidden_size=config['model_params']['hidden_size'])
    
    load = True
    try:
        ckpts = [f for f in os.listdir(log_dir) if f.startswith("step_") and f.endswith(".t7")]
        iters = sorted([int(f.split('_')[-1].split('.')[0]) for f in ckpts])[-1]
    except:
        iters = 0
        load = False
    
    optimizer = AdamW(bert.parameters(), lr=config.get('learning_rate', 1e-4))
    
    accelerator = Accelerator(mixed_precision=config['mixed_precision'], split_batches=True, kwargs_handlers=[ddp_kwargs])
    
    if load:
        checkpoint = torch.load(os.path.join(log_dir, f"step_{iters}.t7"), map_location='cpu')
        state_dict = checkpoint['net']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') # remove `module.`
            new_state_dict[name] = v

        bert.load_state_dict(new_state_dict, strict=False)
        accelerator.print(f'Checkpoint loaded from step {iters}.')
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
    
    bert, optimizer, train_loader = accelerator.prepare(
        bert, optimizer, train_loader
    )

    accelerator.print('Start training...')

    running_loss = 0
    
    while iters < num_steps:
        for batch in train_loader:        
            curr_steps += 1
            
            words, labels, phonemes, input_lengths, masked_indices = batch
            text_mask = length_to_mask(input_lengths)
            
            tokens_pred, words_pred = bert(phonemes, attention_mask=(~text_mask).int())
            
            loss_vocab = 0
            for _s2s_pred, _text_input, _text_length, _masked_indices in zip(words_pred, words, input_lengths, masked_indices):
                target_len = min(_s2s_pred.size(0), _text_input.size(0), _text_length.item())
                loss_vocab += criterion(_s2s_pred[:target_len], _text_input[:target_len])
            loss_vocab /= words.size(0)
            
            loss_token = 0
            sizes = 0
            for _s2s_pred, _text_input, _text_length, _masked_indices in zip(tokens_pred, labels, input_lengths, masked_indices):
                if len(_masked_indices) > 0:
                    m_idx = _masked_indices[_masked_indices < _text_length]
                    if len(m_idx) > 0:
                        loss_token += criterion(_s2s_pred[m_idx], _text_input[m_idx]) 
                        sizes += 1
            
            if sizes > 0:
                loss_token /= sizes
            else:
                loss_token = torch.tensor(0.0, device=accelerator.device)

            loss = loss_vocab + loss_token

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()

            iters = iters + 1
            if iters % log_interval == 0:
                accelerator.print('Step [%d/%d], Loss: %.5f, Vocab Loss: %.5f, Token Loss: %.5f'
                        %(iters, num_steps, running_loss / log_interval, loss_vocab, loss_token))
                running_loss = 0
                
            if iters % save_interval == 0:
                accelerator.print('Saving..')
                state = {
                    'net':  accelerator.get_state_dict(bert),
                    'step': iters,
                    'optimizer': optimizer.state_dict(),
                }
                accelerator.save(state, os.path.join(log_dir, f'step_{iters}.t7'))

            if iters >= num_steps:
                return 

if __name__ == "__main__":
    from accelerate import notebook_launcher
    num_processes = torch.cuda.device_count() if torch.cuda.is_available() else 1
    notebook_launcher(train, args=(), num_processes=num_processes, use_port=33389)
