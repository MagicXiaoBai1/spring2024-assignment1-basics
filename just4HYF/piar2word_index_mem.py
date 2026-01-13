import regex as re


GPT2_TOKENIZER_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

with open("./data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as file:
    text = file.read()
    text_segments = re.findall(GPT2_TOKENIZER_REGEX, text)

