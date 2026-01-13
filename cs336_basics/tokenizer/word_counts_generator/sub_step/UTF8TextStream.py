from typing import Iterator, Iterable


def is_chunk_complete_utf8(chunk: bytes) -> bool:
    try:
        chunk.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False

class UTF8TextStream:
    """将 bytes 块流转换为 UTF-8 文本块流（自动处理跨块字符）"""

    def __init__(self, byte_chunks: Iterable[bytes]):
        self.byte_chunks = byte_chunks

    def __iter__(self) -> Iterator[str]:
        leftover_bytes = b""
        for chunk in self.byte_chunks:
            # 拼接残片
            data = leftover_bytes + chunk

            complete_end = len(data)
            while complete_end > 0 and not is_chunk_complete_utf8(data[:complete_end]):
                complete_end -= 1

            if complete_end == 0:
                # 全部无效？跳过（极端情况）
                text = ""
                leftover_bytes = data
            else:
                text = data[:complete_end].decode("utf-8", errors="ignore")
                leftover_bytes = data[complete_end:]

            if text:
                yield text

        # 处理最后残片（即使不完整也 decode with replace）
        if leftover_bytes:
            yield leftover_bytes.decode("utf-8", errors="replace")