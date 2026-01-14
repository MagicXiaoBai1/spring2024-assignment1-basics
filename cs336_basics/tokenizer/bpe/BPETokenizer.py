
from typing import Iterator, Iterable
from concurrent.futures import ThreadPoolExecutor

from cs336_basics.tokenizer.text_inputStream import InputStreamFromString, InputStreamFromStringIter


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.bytes2token_id = {v: k for k, v in vocab.items()}
        max_token_id = max(vocab.keys(), default=-1)
        self.special_tokens = special_tokens or []

        for token in self.special_tokens:
            if token.encode('utf-8') not in self.bytes2token_id:
                max_token_id += 1
                self.bytes2token_id[token.encode('utf-8')] = max_token_id

        self.max_token_bytes_size = max(len(token) for token in vocab.values())

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        pass

    def encode(self, text: str) -> list[int]:
        return list(self._encode_tokens_stream(InputStreamFromString(text, self.special_tokens)))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return self._encode_tokens_stream(InputStreamFromStringIter(iterable, self.special_tokens))

    def _encode_tokens_stream(self, stream: Iterable[str]) -> Iterator[int]:
        with ThreadPoolExecutor(max_workers=22) as ex:
            # ex.map 会按输入顺序产出结果（即使内部并行）
            for encoded in ex.map(self._encode_one_token2, stream, chunksize=128):
                yield from encoded
        pass

    def _encode_one_token(self, word: str) -> list[int]:
        """
        更快更现代，但与gpt2的行为不符合
        """
        res = []
        utf8_word = word.encode('utf-8')
        now_encoding_start_index = 0
        while now_encoding_start_index < len(utf8_word):
            done_index = now_encoding_start_index + 1
            for now_encoding_end_index in range(min(now_encoding_start_index + self.max_token_bytes_size, len(utf8_word)), now_encoding_start_index, -1):

                now_encode_slice = utf8_word[now_encoding_start_index:now_encoding_end_index]

                if now_encode_slice in self.bytes2token_id:
                    res.append(self.bytes2token_id[now_encode_slice])
                    done_index = now_encoding_end_index
                    break
            now_encoding_start_index = done_index

        return res

    def _encode_one_token2(self, word: str) -> list[int]:
        if word in self.special_tokens:
            return [self.bytes2token_id[word.encode('utf-8')]]
        utf8_word = []
        for i in word.encode('utf-8'):
            utf8_word.append(bytes([i]))

        is_can_merage = len(utf8_word) > 1
        while is_can_merage and len(utf8_word) > 1:
            is_can_merage = False
            for merage_a, merage_b in self.merges:
                for i in range(len(utf8_word) - 2, -1, -1):
                    if utf8_word[i] == merage_a and utf8_word[i + 1] == merage_b:
                        utf8_word[i] = utf8_word[i] + utf8_word[i + 1]
                        del utf8_word[i + 1]
                        is_can_merage = True
        res = []
        for bytes1 in utf8_word:
            res.append(self.bytes2token_id[bytes1])
        return res

    def decode(self, ids: list[int]) -> str:
        res = b""
        with ThreadPoolExecutor(max_workers=8) as ex:
            # ex.map 会按输入顺序产出结果（即使内部并行）
            for decoded in ex.map(lambda x: self.vocab[x], ids, chunksize=128):
                res = res + decoded
        return res.decode('utf-8', errors='replace')