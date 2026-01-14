from typing import Iterator, Iterable
import heapq

from cs336_basics.tokenizer.text_inputStream import InputStreamFromString, InputStreamFromStringIter
from concurrent.futures import ProcessPoolExecutor


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = {(a, b): idx for idx, (a, b) in enumerate(merges)}
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
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque
        
        with ThreadPoolExecutor(max_workers=22) as executor:
            futures_queue = deque()
            max_queue_size = 220  # 控制同时提交的任务数，避免内存占用过大
            
            stream_iter = iter(stream)
            exhausted = False
            
            while True:
                # 填充队列到最大容量
                while len(futures_queue) < max_queue_size and not exhausted:
                    try:
                        token = next(stream_iter)
                        future = executor.submit(self._encode_one_token2, token)
                        futures_queue.append(future)
                    except StopIteration:
                        exhausted = True
                        break
                
                # 如果队列为空且输入已耗尽，结束
                if not futures_queue:
                    break
                
                # 按顺序取出最早提交的结果并yield
                future = futures_queue.popleft()
                yield from future.result()

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

        while len(utf8_word) > 1:
            # 找合并方案
            merage_idx = -1
            merage_order = float('inf')
            for i in range(len(utf8_word) - 1):
                now_pair = (utf8_word[i], utf8_word[i + 1])
                now_pair_order = self.merges.get(now_pair, float('inf'))
                if now_pair_order < merage_order:
                    merage_order = now_pair_order
                    merage_idx = i
            
            if merage_idx != -1:
                utf8_word[merage_idx] = utf8_word[merage_idx] + utf8_word[merage_idx + 1]
                del utf8_word[merage_idx + 1]
            else:
                break
            
        res = []
        for bytes1 in utf8_word:
            res.append(self.bytes2token_id[bytes1])
        return res

    def decode(self, ids: list[int]) -> str:
        res = b""
        for decoded in map(lambda x: self.vocab[x], ids):
            res = res + decoded
        return res.decode('utf-8', errors='replace')