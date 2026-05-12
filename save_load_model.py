import torch

from simpleGPT import SimpleGPTConfig, SimpleGPT

from block import BlockConfig

from tokenizer import _encode_decode_from_vocab_and_merges


def save_model(model, config, vocab, checkpoint_path, bpe_merges):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "vocab_size": len(vocab),
        "block_config": config.block_config,
        "n_blocks": config.n_blocks,
        "device": config.device,
        "vocab": vocab,
        "bpe_merges": bpe_merges,
    }
    torch.save(ckpt, checkpoint_path)
    print(f"[save_model] Model saved to {checkpoint_path}")

def load_model(checkpoint_path, device=None):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = device or ckpt["device"]
    config = SimpleGPTConfig(
        block_config = ckpt["block_config"],
        vocab_size=ckpt["vocab_size"],
        n_blocks=ckpt["n_blocks"],
        device=device
    )
    model = SimpleGPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    vocab = ckpt["vocab"]
    merges = ckpt.get("bpe_merges")
    encode, decode = _encode_decode_from_vocab_and_merges(vocab, merges)
    # Returning vocab and merges as well so we can later save the loaded model again
    return model, config, encode, decode, vocab, merges