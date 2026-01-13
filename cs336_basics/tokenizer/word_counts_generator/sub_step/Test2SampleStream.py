from typing import Iterator, Iterable


def is_chunk_complete_utf8(chunk: bytes) -> bool:
    try:
        chunk.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False

class Test2SampleStream:
    """将 bytes 块流转换为 UTF-8 文本块流（自动处理跨块字符）"""

    def __init__(self, text_chunks: Iterable[str], endingToken: str = "<|endoftext|>"):
        self.endingToken = endingToken
        self.text_chunks = text_chunks

    def __iter__(self) -> Iterator[str]:
        buffer = ""
        pattern = self.endingToken

        # KMP helpers
        def build_lps(p: str):
            lps = [0] * len(p)
            length = 0
            i = 1
            while i < len(p):
                if p[i] == p[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            return lps

        # 移除全缓冲扫描，改为跨块增量KMP
        lps = build_lps(pattern)
        j = 0  # 当前已匹配的pattern前缀长度
        for chunk in self.text_chunks:
            for ch in chunk:
                # KMP状态推进
                while j > 0 and ch != pattern[j]:
                    j = lps[j - 1]
                if ch == pattern[j]:
                    j += 1

                buffer += ch

                # 完成一次匹配：按分隔符切段并yield
                if j == len(pattern):
                    seg = buffer[:-len(pattern)]
                    if seg:
                        yield seg
                        yield "<|endoftext|>"
                    buffer = ""
                    j = lps[j - 1]  # 支持重叠匹配继续

        # 处理最后残片
        if buffer:
            yield buffer
            yield "<|endoftext|>"
            