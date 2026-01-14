from typing import Iterator, Iterable

from cs336_basics.tokenizer.text_inputStream.GPT2TokenStream import GPT2TokenStream
from cs336_basics.tokenizer.text_inputStream.MMapReaderStream import FileReaderStream
from cs336_basics.tokenizer.text_inputStream.Text2SampleStream import Text2SampleStream
from cs336_basics.tokenizer.text_inputStream.UTF8DecodeStream import UTF8DecodeStream


class FullInputStream:
    def __init__(self, file_path: str, special_tokens: list[str] = None):
        if special_tokens is None:
            special_tokens = []
        # reader = MMapReaderStream(file_path)
        self.reader = FileReaderStream(file_path)
        self.utf8_stream = UTF8DecodeStream(self.reader)
        self.test2_sample_stream = Text2SampleStream(self.utf8_stream)
        self.token_stream = GPT2TokenStream(self.test2_sample_stream, special_tokens=special_tokens)

    def __iter__(self) -> Iterator[str]:
        return iter(self.token_stream)

    def get_total_bytes(self) -> int:
        return self.reader.total_bytes
    def get_now_bytes(self) -> int:
        return self.reader.now_bytes

class InputStreamFromString(Iterable[str]):
    def __init__(self, input_string: str, special_tokens: list[str] = None):
        self.token_stream = GPT2TokenStream([input_string], special_tokens=special_tokens)

    def __iter__(self) -> Iterator[str]:
        return iter(self.token_stream)

class InputStreamFromStringIter(Iterable[str]):
    def __init__(self, input_itr: Iterable[str], special_tokens: list[str] = None):
        self.test2_sample_stream = Text2SampleStream(input_itr)
        
        self.token_stream = GPT2TokenStream(self.test2_sample_stream, special_tokens=special_tokens)

    def __iter__(self) -> Iterator[str]:
        return iter(self.token_stream)