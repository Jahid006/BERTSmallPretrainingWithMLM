from glob import glob
from torch.utils.data import DataLoader
import torch

from tiny_bert_config import TinyBertConfig

from tiny_bert.tiny_bert_data_utils import TinyBertTDataset
from tiny_bert.tiny_bert_modeling import TinyBert, TinyBertLM
from tiny_bert.tiny_bert_trainer import TinyBertTrainer
from tiny_bert.tiny_bert_optim import ScheduledOptim

from tokenization.text_processor import Tokenizer


def get_tokenizer(cfg):
    tokenizer = Tokenizer(
        sentencepiece_path=cfg.tokenizer_path,
        max_len=cfg.max_len,
        vocab_size=cfg.vocab_size,
    )
    tokenizer.load()
    tokenizer.initialize()

    return tokenizer


def get_text_data(paths):
    paths = glob(f"{paths}**.txt", recursive=True)
    print(paths)
    textlines = []
    for path in paths:
        textlines += [
            line for line in open(path, "r").readlines()
            if len(line.split(' ')) > 3
        ]
    print(f"Total Lines: {len(textlines)}")

    return textlines


def train(cfg: TinyBertConfig):
    # cfg = TinyBertConfig()
    device = torch.device(cfg.device)

    tokenizer = get_tokenizer(cfg)
    text_data = get_text_data(cfg.train_data_path)

    train_data = TinyBertTDataset(
        text_data, seq_len=cfg.max_len, tokenizer=tokenizer, device=device
    )

    train_loader = DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True, prefetch_factor=2
    )
    bert_base = TinyBert(
        vocab_size=len(tokenizer.vocab),
        d_model=cfg.d_model,
        n_layers=cfg.n_encoder_layer,
        heads=cfg.n_head,
    ).to(device)
    bert_lm = TinyBertLM(bert_base, len(tokenizer.vocab)).to(device)
    bert_trainer = TinyBertTrainer(
        bert_lm, train_loader, device="cuda", schedular=ScheduledOptim
    )

    for epoch in range(cfg.epoch):
        bert_trainer.train(epoch)

    torch.save({"model": bert_lm.state_dict()}, cfg.saving_dir)


if __name__ == "__main__":
    train(TinyBertConfig())
