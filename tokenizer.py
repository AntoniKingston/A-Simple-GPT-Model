import re
from collections import Counter
from typing import Callable, List, Tuple

Merge = Tuple[Tuple[str, str], str]
def _most_common_consecutive_pair(tokens: List[str]) -> Merge:
    counter = Counter(zip(tokens[:-1], tokens[1:]))
    most_common, _ = counter.most_common(1)[0]
    return most_common, most_common[0] + most_common[1]

def _apply_merge(tokens: List[str], merge: Merge) -> List[str]:
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and "".join(tokens[i:i+2]) == merge[1]:
            tokens[i:i+2] = [merge[1]]
        # Advance always by 1 since if merge is applied the list gets shortened by 1
        i+=1
    return tokens

def _encode_decode_from_vocab_and_merges(vocab: List[str], merges: List[Merge]) -> Tuple[Callable[[str], List[int]], Callable[[List[int]], str]]:
    stoi = {token: i for i, token in enumerate(vocab)}
    itos = {i: token for i, token in enumerate(vocab)}

    def encode(text: str) -> List[int]:
        tokens = list(text)
        for merge in merges:
            tokens = _apply_merge(tokens, merge)
        return [stoi[token] for token in tokens]

    def decode(tokens: List[int]) -> str:
        return "".join([itos[token] for token in tokens])
    return encode, decode
def bpe_tokenizer(text: str, num_merges: int, info_interval=100) -> Tuple[Callable[[str], List[int]], Callable[[List[int]], str], List[str], List[Merge]]:
    tokens = list(text)
    merges : List[Merge] = []

    for i in range(num_merges):
        if info_interval and i % info_interval == 0:
            print(f"Learnt {i} merges out of {num_merges}")
        merge = _most_common_consecutive_pair(tokens)
        merges.append(merge)
        tokens = _apply_merge(tokens, merge)

    # # Both initial tokens and those learnt by merges are included in vocab
    vocab = list(dict.fromkeys(["<BOS>", "<EOS>"] + sorted(list(set(tokens) | set(list(text))))))
    # vocab = sorted(list(set(tokens) | set(list(text))))

    encode, decode = _encode_decode_from_vocab_and_merges(vocab, merges)

    return encode, decode, vocab, merges

