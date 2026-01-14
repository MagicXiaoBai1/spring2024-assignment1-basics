import regex as re
from typing import Iterable
from typing import Iterator

GPT2_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class GPT2TokenStream:
    """将文本块流按 GPT-2 规则切分为 token 流"""

    def __init__(self, text_chunks: Iterable[str], special_tokens: list[str]):
        self.text_chunks = text_chunks
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.special_tokens_set = set(special_tokens)
        

    def __iter__(self) -> Iterator[str]:
        for chunk in self.text_chunks:
            if len(self.special_tokens) == 0:
                sub_chunks = [chunk]
            else:
                sub_chunks = re.split(f"({'|'.join(map(re.escape, self.special_tokens))})", chunk)
            for sub_chunk in sub_chunks:
                if sub_chunk in self.special_tokens_set:
                    yield sub_chunk
                    continue
                # 分词并 yield
                for match in re.finditer(GPT2_REGEX, sub_chunk):
                    token = match.group()
                    if token:  # 跳过空字符串
                        yield token