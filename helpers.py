import torch

from simpleGPT import SimpleGPTConfig, SimpleGPT

from block import BlockConfig

from tokenizer import _encode_decode_from_vocab_and_merges

def save_bpe_tokenization(vocab, bpe_merges, checkpoint_path):
    torch.save(
        {
            "vocab": vocab,
            "bpe_merges": bpe_merges,
        },
        checkpoint_path,
    )

def load_bpe_tokenization(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    merges = ckpt["bpe_merges"]
    encode, decode = _encode_decode_from_vocab_and_merges(vocab, merges)
    return encode, decode, vocab, merges
def save_encoding(encoding, checkpoint_path):
    torch.save(encoding, checkpoint_path)

def load_encoding(checkpoint_path):
    return torch.load(checkpoint_path)

def save_model(model, config, checkpoint_path):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "block_config": config.block_config,
        "n_blocks": config.n_blocks,
        "vocab_size": config.vocab_size,
    }
    torch.save(ckpt, checkpoint_path)
    print(f"[save_model] Model saved to {checkpoint_path}")

def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = SimpleGPTConfig(
        block_config=ckpt["block_config"],
        vocab_size=ckpt["vocab_size"],
        n_blocks=ckpt["n_blocks"],
    )
    model = SimpleGPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, config

def split_encoding(encoding: torch.Tensor, tr_val_split: int = 0.9):
    return (
        torch.tensor(encoding[:int(len(encoding) * tr_val_split)]),
        torch.tensor(encoding[int(len(encoding) * tr_val_split):]),
    )

