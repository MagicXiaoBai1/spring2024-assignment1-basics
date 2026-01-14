from typing import Iterator, Iterable
import regex as re

class Text2SampleStream:

    def __init__(self, text_chunks: Iterable[str]):
        self.text_chunks = text_chunks
        self.split_pattern = re.compile(r"\S\s")

    def __iter__(self) -> Iterator[str]:
        buffer = ""
        for chuck in self.text_chunks:
            for char in chuck:
                if len(buffer) > 0 and self.split_pattern.match(buffer[-1] + char):
                    yield buffer
                    buffer = ""                
                buffer += char
                
        # 处理最后残片
        if buffer:
            yield buffer
            