

import logging
import os
import pickle

from cs336_basics.tokenizer.bpe.bpe_trainer import BPETrainer
from cs336_basics.tokenizer.word_counts_generator.word_counts_generator import generate_word_counts
logger = logging.getLogger(__name__)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
):
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path: str | os.PathLike
            Path to BPE tokenizer training data.
        vocab_size: int
            Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: list[str]
            A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        Tuple of (vocab, merges):
            vocab: dict[int, bytes]
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    logger.info(f"Starting BPE training on {input_path}...")
    generate_word_counts(input_path, "output/tmp.pkl", special_tokens)
    bpe_trainer = BPETrainer(vocab_size, special_tokens)
    
    with open("output/tmp.pkl", "rb") as f:
        word_counts = pickle.load(f)
    
    return bpe_trainer.train(word_counts)
    
    

if __name__ == "__main__":
    pass
