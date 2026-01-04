"""
Configuration and logging utilities
"""
import os
import yaml
from datetime import datetime


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config, local_rank, world_size):
    """Setup logging files and directories"""
    is_main_process = (local_rank == 0)
    
    if not is_main_process:
        return None, None
    
    # Extract config values
    log_dir = config['logging']['log_dir']
    save_dir = config['logging']['save_dir']
    batch_size = config['training']['batch_size']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    lr = config['training']['learning_rate']
    mlm_prob = config['training']['mlm_prob']
    model_config = config['model']
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    train_log_file = os.path.join(log_dir, f"train_{timestamp}.txt")
    val_log_file = os.path.join(log_dir, f"val_{timestamp}.txt")
    
    # Write training log header
    with open(train_log_file, "w") as f:
        f.write(f"Training Log - Started at {datetime.now()}\n")
        f.write(f"Config file: config.yaml\n")
        f.write(f"Batch size: {batch_size}, Grad accum: {gradient_accumulation_steps}\n")
        f.write(f"Effective batch: {batch_size * world_size * gradient_accumulation_steps}\n")
        f.write(f"Learning rate: {lr}, MLM prob: {mlm_prob}\n")
        f.write(f"Model: hidden={model_config['hidden_size']}, layers={model_config['num_layers']}, heads={model_config['num_heads']}\n")
        f.write("="*80 + "\n")
        f.write("Step\tLoss\tMLM_Loss\tCTC_Loss\tMLM_Acc\tWER\tF1\tPrecision\tRecall\n")
    
    # Write validation log header
    with open(val_log_file, "w") as f:
        f.write(f"Validation Log - Started at {datetime.now()}\n")
        f.write(f"Config file: config.yaml\n")
        f.write("="*80 + "\n")
        f.write("Step\tLoss\tMLM_Loss\tCTC_Loss\tMLM_Acc\tWER\tF1\tPrecision\tRecall\n")
    
    return train_log_file, val_log_file


def log_training_step(log_file, global_step, metrics):
    """Log training metrics to file"""
    with open(log_file, "a") as f:
        f.write(f"{global_step}\t{metrics['total_loss']:.6f}\t{metrics['mlm_loss']:.6f}\t"
                f"{metrics['ctc_loss']:.6f}\t{metrics['mlm_acc']:.6f}\t{metrics['wer']:.6f}\t"
                f"{metrics['f1']:.6f}\t{metrics['precision']:.6f}\t{metrics['recall']:.6f}\n")


def log_validation_step(log_file, global_step, metrics):
    """Log validation metrics to file"""
    with open(log_file, "a") as f:
        total_loss = metrics['mlm_loss'] + metrics['ctc_loss']
        f.write(f"{global_step}\t{total_loss:.6f}\t{metrics['mlm_loss']:.6f}\t"
                f"{metrics['ctc_loss']:.6f}\t{metrics['mlm_acc']:.6f}\t{metrics['wer']:.6f}\t"
                f"{metrics['f1']:.6f}\t{metrics['precision']:.6f}\t{metrics['recall']:.6f}\n")


def log_completion(train_log_file, val_log_file, global_step):
    """Log training completion"""
    with open(train_log_file, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Training completed at {datetime.now()}\n")
        f.write(f"Final step: {global_step}\n")
    
    with open(val_log_file, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Training completed at {datetime.now()}\n")
