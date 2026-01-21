# PL-BERT-BPE

## Installation

### 1. Install espeak-ng

**Mac/Linux (Debian/Ubuntu):**
```bash
sudo apt-get install espeak-ng
```

**Mac (Homebrew):**
```bash
brew install espeak-ng
```

**Windows:**
Download and install from [eSpeak NG Releases](https://github.com/espeak-ng/espeak-ng/releases)

### 2. Install uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```


### 3. Activate Virtual Environment

**Mac/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv/Scripts/activate
```

### 4. Install Project Dependencies

Choose one of the following:

```bash
uv pip install -e .
```

Or:

```bash
uv sync
```

## Setup Instructions

### 1. Download Wikipedia ID Dataset

Download the Indonesian Wikipedia dataset from:
```
https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.id
```

### 2. Preprocess Data

Run the preprocessing script to prepare your data.

### 3. Configure Vocabulary (Optional)

Set phoneme vocabulary and pruned BPE vocabulary if needed.

### 4. Train the Model

Run the multi-GPU training script:

```bash
sh train_multi_gpu.sh
```

### 5. Adjust Model Size (If Needed)

Modify model size in the training configuration as needed.