import os
import json
import random
import glob
import regex
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from utils.add_noise import Synthesizer
from models.word_char_tokenizer import PreWordTokenizer, PreCharTokenizer
from utils.common import SpecialTokens, all_special_tokens

char_tokenizer = PreTrainedTokenizerFast(tokenizer_file="spell_model/char_tokenizer.json")
word_tokenizer = PreTrainedTokenizerFast(tokenizer_file="spell_model/word_tokenizer.json")
# Work around for missing pad_token
char_tokenizer.pad_token = SpecialTokens.pad
word_tokenizer.pad_token = SpecialTokens.pad
pre_char_tokenizer = PreCharTokenizer()
pre_word_tokenizer = PreWordTokenizer()

num_max_word = 192
num_max_char = 16


def get_key(item):
    # item in format: /path/to/corpus_{idx}.txt
    return int(item.rsplit('.', 1)[0].rsplit('_', 1)[1])


pattern = regex.compile(r"""
[^aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆ
fFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRs
StTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'()*+,-.
/:;<=>?@[\]^_`{|}~[:space:]]""")


def custom_collator(batch):
    """
    Collator for misspelled dataset
    Args:
        batch: list[success], list[origin_tokens], list[synth_tokens], list[onehot_labels]
    Returns:
        Dict {
            "word_input_ids": torch.LongTensor of shape batch x word_seq_length
            "word_attention_mask": torch.LongTensor of shape batch x word_seq_length
            "word_token_type_ids": torch.LongTensor of shape batch x word_seq_length
            "char_input_ids": torch.LongTensor of shape (batch x word_seq_length) x char_sequence_length
            "char_attention_mask": torch.LongTensor of shape (batch x word_seq_length) x char_sequence_length
            "char_token_type_ids": torch.LongTensor of shape (batch x word_seq_length) x char_sequence_length
            "correction_labels": torch.LongTensor of shape batch x word_seq_length
            "detection_labels": torch.LongTensor of shape batch x word_seq_length
        }
    """
    _, batch_origin_tokens, batch_synth_tokens, batch_onehot_labels = zip(*batch)

    # Pad batch_onehot_labels to shape B x Seq Len
    max_length = 0
    max_idx = 0
    # +2 accounts for [CLS] and [SEP]
    for idx, item in enumerate(batch_onehot_labels):
        if len(item) + 2 > max_length:
            max_length = len(item) + 2
            max_idx = idx

    batch_onehot_labels = [[0] + item + [0] * (max_length - len(item) - 1) for item in batch_onehot_labels]
    # Truncate to maximum number of words
    batch_onehot_labels = [item[:num_max_word] for item in batch_onehot_labels]
    batch_onehot_labels = torch.LongTensor(batch_onehot_labels)

    # Word/Char encoding
    batch_origin_tokens = [' '.join(tks) for tks in batch_origin_tokens]
    batch_synth_tokens = [' '.join(tks) for tks in batch_synth_tokens]

    # input_ids, token_type_ids, attention_mask
    batch_synth_enc = word_tokenizer(batch_synth_tokens, padding=True, truncation=True,
                                     max_length=num_max_word, return_tensors="pt")

    batch_origin_enc = word_tokenizer(batch_origin_tokens, padding=True, truncation=True,
                                      max_length=num_max_word, return_tensors="pt")
    batch_correction_lbs = batch_origin_enc["input_ids"]

    # Create batch of char ids = batch(sent 1) "stack on" batch(sent 2)
    batch_sent_words = []

    _, seq_word_len = batch_synth_enc["input_ids"].shape
    for sent_idx, synth_tokens in enumerate(batch_synth_tokens):
        synth_tokens = synth_tokens.split()
        synth_tokens = [SpecialTokens.cls] + synth_tokens[:seq_word_len - 2] + [SpecialTokens.sep]
        synth_tokens = synth_tokens + [SpecialTokens.pad] * (seq_word_len - len(synth_tokens))
        for word_idx, word in enumerate(synth_tokens):
            if word in all_special_tokens:
                batch_sent_words.append(SpecialTokens.unk)
            else:
                batch_sent_words.append(word)

    batch_char_tok = [' '.join(pre_char_tokenizer.pre_tokenize(word)) for word in batch_sent_words]
    batch_char_enc = char_tokenizer(batch_char_tok, padding=True, truncation=True,
                                    max_length=num_max_char, return_tensors="pt")

    assert (batch_char_enc["input_ids"].size(0) / len(batch_origin_tokens) == batch_synth_enc["input_ids"].size(1)), \
        f'ERROR {batch_char_enc["input_ids"].size(0)} {len(batch_origin_tokens)} {batch_synth_enc["input_ids"].size(1)}'
    assert (batch_synth_enc["input_ids"].size() == batch_correction_lbs.size() == batch_onehot_labels.size()), \
        f'[ERROR] {batch_synth_enc["input_ids"].size()} {batch_correction_lbs.size()} {batch_onehot_labels.size()}\n' \
        f'{batch_origin_tokens[max_idx]}\n' \
        f'{batch_synth_tokens[max_idx]}'

    batch_onehot_labels[batch_onehot_labels != 0] = 1
    return {
        "word_input_ids": batch_synth_enc["input_ids"],
        "word_attention_mask": batch_synth_enc["attention_mask"],
        "word_token_type_ids": batch_synth_enc["token_type_ids"],
        "char_input_ids": batch_char_enc["input_ids"],
        "char_attention_mask": batch_char_enc["attention_mask"],
        "char_token_type_ids": batch_char_enc["token_type_ids"],
        "correction_labels": batch_correction_lbs,
        "detection_labels": batch_onehot_labels
    }


class MisspelledDataset(Dataset):
    """
    Create a synthesized misspelled dataset
    """

    def __init__(self,
                 corpus_dir: str,
                 percent_err: float = 0.2,
                 min_num_tokens: int = 5):
        """
        Args:
            corpus_dir: directory contains list of corpus_{idx}.txt files
            percent_err: percentage of misspelled tokens to generate
            min_num_tokens: minimum number of tokens
        """
        super().__init__()
        self.corpus_dir = corpus_dir
        self.percent_err = percent_err
        self.min_num_tokens = min_num_tokens
        self.pre_word_tokenizer = pre_word_tokenizer

        stats_file = os.path.join(corpus_dir, "stats.json")
        self.stats = json.load(open(stats_file))

        self.lines_per_file = self.stats["lines_per_file"]
        self.num_lines = self.stats["total_lines"]
        self.num_files = self.stats["num_files"]

        self.corpus_files = glob.glob(os.path.join(self.corpus_dir, "corpus_*.txt"))
        self.corpus_files = sorted(self.corpus_files, key=lambda x: get_key(x))
        self.synthesizer = Synthesizer()
        self.file_size = OrderedDict()

        for idx in range(self.num_files):
            if idx != self.num_files - 1:
                self.file_size[idx] = self.lines_per_file
            else:
                remainder = self.num_lines % self.lines_per_file
                self.file_size[idx] = remainder if remainder > 0 else self.lines_per_file

    def __len__(self):
        return self.num_lines

    def __getitem__(self, index):
        # Simply random file, then random line
        file_idx = random.randint(0, len(self.corpus_files) - 1)
        line_idx = random.randint(0, self.file_size[file_idx] - 1)

        with open(self.corpus_files[file_idx]) as fp:
            for count, line in enumerate(fp):
                if count == line_idx:
                    break

        line = line.replace('\u200b', '')  # Work around for [​] char
        line = pattern.sub("", line).strip()
        if not line:
            # Random and get another sentence
            return self.__getitem__(123)

        origin_tokens = self.pre_word_tokenizer.pre_tokenize(line)
        if len(origin_tokens) < self.min_num_tokens:
            # If the sentence is too short, skip
            return self.__getitem__(123)

        success, origin_tokens, tokens, onehot_label = self.synthesizer.add_noise(
            origin_tokens=origin_tokens, percent_err=self.percent_err)

        if not success:
            # If failed to add noise like case origin_tokens = ["35", "."]
            # We get another sample
            return self.__getitem__(123)

        return success, origin_tokens, tokens, onehot_label


if __name__ == '__main__':
    random.seed(31)
    np.random.seed(12)

    # ds = MisspelledDataset(corpus_dir="/home/local/BM/Datasets/SpellNews")
    # print(ds[123])
    # loader = DataLoader(ds, batch_size=2, collate_fn=custom_collator, drop_last=True)
    #
    # for sample in tqdm(loader, total=len(ds)//2):
    #     pass

    sample_batch = [
        (
            True,
            ['Trong', 'trận', 'đấu', 'thuộc', 'vòng', '19', 'giải', 'V-League', '2017', 'trên', 'sân', 'Long', 'An',
             ',', 'đội', 'chủ', 'nhà', 'một', 'lần', 'nữa', 'lại', 'để', 'chiến', 'thắng', 'tuột', 'khỏi', 'tầm', 'tay',
             'dù', 'đã', 'dẫn', 'trước', 'Sana', 'Khánh', 'Hoa', 'BVN', 'từ', 'rất', 'sớm', '.'],
            ['Trong', 'trận', 'đấu', 'thuộc', 'vong', '19', 'giải', 'V-League', '2017', 'trên', 'sân', 'Long', 'An',
             ',', 'đội', 'chủ', 'nhà', 'một', 'lần', 'nữa', 'lại', 'ddể', 'chiến', 'thắng', 'tuột', 'khỏi', 'tầm',
             'tby', 'ởù', 'đã', 'dẫn', 'truwớc', 'Sana', 'Khyánhy', 'Hoa', 'BVN', 'từ', 'rất', 'sớm', '.'],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 4, 6, 0, 0, 2, 0, 4, 0, 0,
             0, 0, 0, 0]
        ),
        (
            True,
            ['Vì', 'vậy', ',', 'viễn', 'cảnh', 'đối', 'đầu', 'với', '150.000', 'quân', 'Tào', 'vẫn', 'khiến', 'liên',
             'minh', 'Tôn', '-', 'Lưu', '...', 'khá', 'dễ', 'chịu', '.'],
            ['Vì', 'zay', ',', 'viễn', 'cảnh', 'đối', 'đầu', 'với', '150.000', 'quân', 'Tào', 'vặn', 'khriến', 'liên',
             'minh', 'Ton', '-', 'Lưu', '...', 'khá', 'dễ', 'chịu', '.'],
            [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 4, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0]
        )
    ]

    out = custom_collator(sample_batch)
    pass
