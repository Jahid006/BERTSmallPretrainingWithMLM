class TinyBertConfig:
    tokenizer_path = "./tokenization/vocab/vocab_bert.4096"
    vocab_size = 4096
    max_len = 128
    d_model = 256
    n_head = 4
    n_encoder_layer = 4

    batch_size = 128
    epoch = 20
    device = "cuda"

    train_data_path = "./pretraining_data/tiny_bert_pretraining_data_v3"

    saving_dir = "./tiny_bert_artifact/tiny_bert_ner-v3.pt"
