

import logging
import os
import pickle
import numpy as np

from cs336_basics.tokenizer.bpe.BPETokenizer import BPETokenizer
from cs336_basics.tokenizer.bpe.bpe_trainer import BPETrainer
from cs336_basics.tokenizer.text_inputStream import FileReaderStream, UTF8DecodeStream
from cs336_basics.tokenizer.word_counts_generator.word_counts_generator import generate_word_counts
import subprocess
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
    
    
def run_train():
    special_tokens = ["<|endoftext|>"]
    bpe_trainer = BPETrainer(32000, special_tokens)
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    input_path = "/home/love/python_code/spring2024-assignment1-basics/output/owt_train.txt_word_counts_train.pkl"

    with open(input_path, "rb") as f:
        word_counts = pickle.load(f)
    
    res = bpe_trainer.train(word_counts)
    
    with open(input_path.replace(".pkl", "_res.pkl"), "wb") as f:
        pickle.dump(res, f)

def test_encode_iterable_tinystories_matches_tiktoken():
    print(f"HI there! Current working directory: {os.getcwd()}")

    with open("./output/TinyStoriesV2_res.pkl", "rb") as f:
    # with open("./output/owt_train.txt_word_counts_train_res.pkl", "rb") as f:
        tokenizer_param = pickle.load(f)
    tokenizer = BPETokenizer(tokenizer_param[0], tokenizer_param[1], special_tokens=["<|endoftext|>"])
    
    reader = FileReaderStream("./data/TinyStoriesV2-GPT4-train.txt")
    utf8_stream = UTF8DecodeStream(reader)

    res = []
    TIMES = 1000000
    idx = 0
    print("start")
    for _id in tokenizer.encode_iterable(utf8_stream):
        idx += 1
        if idx % TIMES == 0:
            print(f"Processed {idx} tokens")
            with open(f"./output/tinystories_tiktokens{idx}.txt", "wb") as f:
                np.array(res, dtype=np.uint16).tofile(f)
                res = []
        res.append(_id)

# nohup /home/love/miniconda3/envs/dl_env/bin/python ... > output.log 2>&1 &
if __name__ == "__main__":
    # print(subprocess.check_output(['pwd']).decode().strip())
    #
    # with open("./output/TinyStoriesV2_res.pkl", "rb") as f:
    # # with open("./output/owt_train.txt_word_counts_train_res.pkl", "rb") as f:
    #     res = pickle.load(f)
    # print(res[0])
    # print("Done")
    test_encode_iterable_tinystories_matches_tiktoken()
    print("Done")