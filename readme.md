
# Project Name
PreTraining Pipeline for smaller Bert
## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Pretrain TinyBERT](#introduction)
- [Train Tokenizer](#installation)

## Introduction
This repo provides a methodology for pretraining a BERT with Masked Language Modeling Objective. 
## Installation
Download Anaconda and create a conda environment(python 3.10.x) with the requirement.txt file.
```shell
    pip install -r requirements.txt --no-cache
```
## PretrainTinyBERT
1. Change the config file at tiny_bert_config.py to your need
```python 
class TinyBertConfig:
    # your tokenizer path
    tokenizer_path = "./tokenization/vocab/vocab_bert.4096"
    # vocab size
    vocab_size = 4096
    # max sequence length
    max_len = 128
    # embedding dim
    d_model = 256
    #number of head
    n_head = 4
    # number of encoder layers
    n_encoder_layer = 4

    batch_size = 128
    epoch = 20
    device = "cuda"
    train_data_path = "./pretraining_data/tiny_bert_pretraining_data_v3"
    saving_dir = "./tiny_bert_artifact/tiny_bert_ner-v3.pt"
```
2. run ```python pretrain_tiny_bert.py``` from cmd


## TrainTokenizer
To Train, use the following code snippet.

```python
TOKENIZER_PATH = 'where/to/save/my/tokenizer/artifact'
TOKENIZER_MAX_SEQ = 128 # max sequence length your model can handle
TOKENIZER_VOCAB_SIZE = 8192 # number of vacab you want
DATA_PATH = "tokenizer_training_data.txt"

from tokenization.text_processor import Tokenizer
sp_tokenizer = Tokenizer(
    sentencepiece_path=TOKENIZER_PATH,
    max_len=TOKENIZER_MAX_SEQ,
    vocab_size=TOKENIZER_VOCAB_SIZE
)
sp_tokenizer.train(DATA_PATH)
```

