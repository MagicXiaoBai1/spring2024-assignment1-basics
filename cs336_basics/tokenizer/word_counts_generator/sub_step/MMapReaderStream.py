import mmap
import os
from typing import Iterator, BinaryIO, Iterable

class MMapReaderStream(Iterable[bytes]):
    """从大文件 mmap 中按块 yield bytes"""
    def __init__(self, filepath: str, chunk_size: int = 64 * 1024 * 1024):
        self.filepath = filepath
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[bytes]:
        with open(self.filepath, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                size = len(mm)
                offset = 0
                while offset < size:
                    end = min(offset + self.chunk_size, size)
                    yield mm[offset:end]
                    offset = end

class FileReaderStream:
    def __init__(self, filepath: str, chunk_size: int = 64 * 1024 * 1024):  # 64MB
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.total_bytes = os.path.getsize(filepath)
        self.now_bytes = 0

    def __iter__(self) -> Iterator[bytes]:
        with open(self.filepath, "rb") as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                self.now_bytes += len(chunk)
                yield chunk