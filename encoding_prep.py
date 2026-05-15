from tokenizer import bpe_tokenizer
import os
from helpers import save_bpe_tokenization, save_encoding, split_encoding



def encoding_prep(corpus_path, tokenizer_save_dir_path, tokenizer_name, encoding_save_dir_path, encodings_name, version, num_merges=20000, info_interval=1000, tr_val_split: float = 0.9):
    with open(corpus_path, "r") as f:
        corpus = f.read()
    encode, _, vocab, merges = bpe_tokenizer(corpus, num_merges, info_interval)
    save_bpe_tokenization(vocab, merges, os.path.join(tokenizer_save_dir_path, tokenizer_name + "_" + version + ".pt"))
    encoding = encode(corpus)
    tr_encoding, val_encoding = split_encoding(encoding, tr_val_split)
    save_encoding(tr_encoding, os.path.join(encoding_save_dir_path, "tr_" + encodings_name + "_" + version + ".pt"))
    save_encoding(val_encoding, os.path.join(encoding_save_dir_path, "val_" + encodings_name + "_" + version + ".pt"))

if __name__ == "__main__":
    corpus_path = "corpora/mickiewicz_corpus.txt"
    tokenizer_save_dir_path = "tokenizers"
    tokenizer_name = "tokenizer_mickiewicz"
    encoding_save_dir_path = "encodings"
    encodings_name = "encoding_mickiewicz"
    version = "010"
    num_merges = 4000
    info_interval = 1000
    tr_val_split = 0.9
    encoding_prep(corpus_path, tokenizer_save_dir_path, tokenizer_name, encoding_save_dir_path, encodings_name, version, num_merges, info_interval, tr_val_split)