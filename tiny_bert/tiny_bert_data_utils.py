import torch
import random
from torch.utils.data import Dataset
import itertools


class TinyBertTDataset(Dataset):
    def __init__(self, data_pair, tokenizer, device, seq_len=128):
        self.tokenizer = tokenizer
        self.device = device
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, index):
        t1 = self.lines[index]
        t1_random, t1_label = self.random_word(t1)

        t1 = (
            [self.tokenizer.char_to_idx["<cls>"]]
            + t1_random
            + [self.tokenizer.end_token_id]
        )
        t1_label = (
            [self.tokenizer.char_to_idx["<cls>"]]
            + t1_label
            + [self.tokenizer.end_token_id]
        )

        bert_input = t1[: self.seq_len]
        bert_label = t1_label[: self.seq_len]
        padding = [
            self.tokenizer.get_id("<pad>")
            for _ in range(self.seq_len - len(bert_input))
        ]

        bert_input.extend(padding), bert_label.extend(padding)

        output = {"input_ids": bert_input, "label_ids": bert_label}

        return {key: torch.tensor(value).to(self.device) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []

        # 15% of the tokens would be replaced
        for i, token in enumerate(tokens):
            prob = random.random()

            # remove cls and sep token
            token_id = self.tokenizer.tokenize(token)

            if prob < 0.15:
                prob /= 0.15

                # 80% chance change token to mask token
                if prob < 0.8:
                    for i in range(len(token_id)):
                        output.append(self.tokenizer.get_id("<mask>"))

                # 10% chance change token to random token
                elif prob < 0.9:
                    for i in range(len(token_id)):
                        output.append(random.randrange(len(self.tokenizer.vocab)))

                # 10% chance change token to current token
                else:
                    output.append(token_id)

                output_label.append(token_id)

            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)

        # flattening
        output = list(
            itertools.chain(
                *[[x] if not isinstance(x, list) else x for x in output])
        )
        output_label = list(
            itertools.chain(
                *[[x] if not isinstance(x, list) else x for x in output_label]
            )
        )
        assert len(output) == len(output_label)
        return output, output_label
